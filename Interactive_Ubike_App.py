import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate, Bidirectional, BatchNormalization, Attention, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from datetime import datetime, timedelta
import tensorflow as tf
import os

NUMERIC_FEATURES = ['rent', 'return', 'times used', 'net rent']

# 支援 matplotlib 顯示中文 (Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 解決 Keras 版本不相容問題，並加入 Huber Loss
custom_objects = {"mse": tf.keras.losses.MeanSquaredError(), "Huber": Huber()}

class ProgressEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.stopped_epoch > 0:
            return # 已經觸發早停，交由父類別印出結束訊息
        if self.wait > 0:
            print(f" --- [早停機制進度]: 目前連續 {self.wait}/{self.patience} 回合無進步")
        else:
            print(f" --- [早停機制進度]: 驗證集損失下降！計數歸零 (0/{self.patience})")

def preprocess_dataframe(df):
    print("正在執行資料前處理與時間序列補全 (解決因無人借還導致的時間跳號問題)...")
    # 將年月日小時轉換為 datetime
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    continuous_dfs = []
    unique_stations = df['ID'].unique()
    total = len(unique_stations)
    
    for step, station_id in enumerate(unique_stations):
        print(f"\r補全進度: {step+1}/{total} 站點", end="", flush=True)
        station_df = df[df['ID'] == station_id].set_index('time')
        station_df = station_df[~station_df.index.duplicated(keep='first')]
        # 建立完整的時間序列，確保序列連續性 (無人借還的時段自動補齊並填 0)
        idx = pd.date_range(start=station_df.index.min(), end=station_df.index.max(), freq='h')
        station_df = station_df.reindex(idx, fill_value=0)
        station_df['ID'] = station_id
        station_df.reset_index(names='time', inplace=True)
        continuous_dfs.append(station_df)
        
    print()
    new_df = pd.concat(continuous_dfs, ignore_index=True)
    
    new_df['rent'] = new_df['rent'].fillna(0).astype(int)
    new_df['return'] = new_df['return'].fillna(0).astype(int)

    print("正在進行極端值截斷 (Outlier Clipping)...")
    # 計算每個站點的 99 百分位數，作為異常突發流量的上限
    p99 = new_df.groupby('ID')[['rent', 'return']].transform('quantile', 0.99)
    # 將超過 99 百分位數的極端異常值壓平，保護 MinMaxScaler 不被壓縮
    new_df['rent'] = np.where(new_df['rent'] > p99['rent'], p99['rent'], new_df['rent'])
    new_df['return'] = np.where(new_df['return'] > p99['return'], p99['return'], new_df['return'])
    
    # 根據截斷後的數據重新計算衍生特徵
    new_df['times used'] = new_df['rent'] + new_df['return']
    new_df['net rent'] = new_df['rent'] - new_df['return']

    # 重新計算所需時間特徵
    new_df['year'] = new_df['time'].dt.year
    new_df['month'] = new_df['time'].dt.month
    new_df['day'] = new_df['time'].dt.day
    new_df['hour'] = new_df['time'].dt.hour
    new_df['day_of_week'] = new_df['time'].dt.dayofweek
    
    # 平滑的時間特徵
    new_df['hour_sin'] = np.sin(2 * np.pi * new_df['hour'] / 24)
    new_df['hour_cos'] = np.cos(2 * np.pi * new_df['hour'] / 24)
    new_df['day_sin'] = np.sin(2 * np.pi * new_df['day_of_week'] / 7)
    new_df['day_cos'] = np.cos(2 * np.pi * new_df['day_of_week'] / 7)
    
    # 尖銳的時間狀態標籤 (Binary Features)
    new_df['is_weekend'] = (new_df['day_of_week'] >= 5).astype(float)
    new_df['is_morning_peak'] = new_df['hour'].isin([7, 8, 9]).astype(float)  # 早上上班上課尖峰
    new_df['is_evening_peak'] = new_df['hour'].isin([17, 18, 19]).astype(float) # 傍晚下班下課尖峰
    
    return new_df

def build_dataset(data_df, le, scalers=None, seq_length=24):
    X_seq, X_id, y = [], [], []
    is_training = (scalers is None)
    if is_training:
        scalers = {}
        
    print("正在建立模型輸入序列...")
    unique_stations = data_df['station_idx'].unique()
    total_stations = len(unique_stations)
    for step, idx in enumerate(unique_stations): 
        print(f"\r處理進度: {step+1}/{total_stations} 站點 ({(step+1)/total_stations*100:.1f}%)", end="", flush=True)
        group = data_df[data_df['station_idx'] == idx].sort_values('time') 
        
        # 使用 4 個數值特徵
        if is_training:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(group[NUMERIC_FEATURES])
            scalers[idx] = scaler
        else:
            scaler = scalers[idx]
            scaled_values = scaler.transform(group[NUMERIC_FEATURES])
            
        # 7 個時間狀態特徵
        time_feats = group[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'is_morning_peak', 'is_evening_peak']].values
        combined_data = np.hstack([scaled_values, time_feats])

        for i in range(len(combined_data) - seq_length):
            X_seq.append(combined_data[i : i + seq_length])
            X_id.append(idx)
            # y 只需要預測 rent 和 return (即前兩個數值特徵)
            y.append(scaled_values[i + seq_length, :2])
            
    print() # 換行
    return np.array(X_seq), np.array(X_id), np.array(y), scalers

def show_evaluation_plots(y_test, y_pred_scaled, X_id_test, le, all_scalers):
    y_test_inv = np.zeros_like(y_test)
    y_pred_inv = np.zeros_like(y_pred_scaled)
    
    # 針對每一個站點批次進行反標準化
    for s_idx in np.unique(X_id_test):
        mask = (X_id_test == s_idx)
        scaler = all_scalers[s_idx]
        
        # 因為 scaler 是用 4 個特徵 fit 的，需要補齊 4 個維度才能反標準化
        y_test_padded = np.zeros((np.sum(mask), 4))
        y_test_padded[:, :2] = y_test[mask]
        y_test_padded_df = pd.DataFrame(y_test_padded, columns=NUMERIC_FEATURES)
        
        y_pred_padded = np.zeros((np.sum(mask), 4))
        y_pred_padded[:, :2] = y_pred_scaled[mask]
        y_pred_padded_df = pd.DataFrame(y_pred_padded, columns=NUMERIC_FEATURES)
        
        y_test_inv[mask] = scaler.inverse_transform(y_test_padded_df)[:, :2]
        y_pred_inv[mask] = scaler.inverse_transform(y_pred_padded_df)[:, :2]

    print("顯示模型評估可視化圖表 (關閉圖表後繼續)...")
    
    # 1. 實際值 vs 預測值 散佈圖 (Scatter Plot)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_inv[:, 0], y_pred_inv[:, 0], alpha=0.3, color='blue', s=10)
    max_rent = max(np.max(y_test_inv[:, 0]), np.max(y_pred_inv[:, 0]))
    plt.plot([0, max_rent], [0, max_rent], 'r--', lw=2)
    plt.title('Rent: Actual vs Predicted (租借量：實際 vs 預測)')
    plt.xlabel('Actual Rent (實際租借量)')
    plt.ylabel('Predicted Rent (預測租借量)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_inv[:, 1], y_pred_inv[:, 1], alpha=0.3, color='green', s=10)
    max_return = max(np.max(y_test_inv[:, 1]), np.max(y_pred_inv[:, 1]))
    plt.plot([0, max_return], [0, max_return], 'r--', lw=2)
    plt.title('Return: Actual vs Predicted (歸還量：實際 vs 預測)')
    plt.xlabel('Actual Return (實際歸還量)')
    plt.ylabel('Predicted Return (預測歸還量)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    # 2. 測試集中特定站點的時間序列預測圖
    target_id = 19082 
    if target_id not in le.classes_:
        target_id = le.inverse_transform([np.unique(X_id_test)[0]])[0]
        
    s_idx_target = le.transform([target_id])[0]
    indices = np.where(X_id_test == s_idx_target)[0]
    
    if len(indices) > 0:
        num_hours = min(100, len(indices)) # 取最後100小時
        
        y_true_target = y_test_inv[indices][-num_hours:]
        y_pred_target = y_pred_inv[indices][-num_hours:]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # 租借量 (Rent)
        ax1.plot(y_true_target[:, 0], label='Actual Rent (實際租借)', alpha=0.7, color='blue', marker='.')
        ax1.plot(y_pred_target[:, 0], label='Pred Rent (預測租借)', linestyle='--', color='red', marker='.')
        ax1.set_title(f'Station {target_id} - Test Set Rent Prediction (最後 {num_hours} 小時測試集預測)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 歸還量 (Return)
        ax2.plot(y_true_target[:, 1], label='Actual Return (實際歸還)', alpha=0.7, color='green', marker='.')
        ax2.plot(y_pred_target[:, 1], label='Pred Return (預測歸還)', linestyle='--', color='orange', marker='.')
        ax2.set_title(f'Station {target_id} - Test Set Return Prediction (最後 {num_hours} 小時測試集預測)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


def train_model():
    print("\n--- 開始訓練模型與產生可視化結果 ---")
    if not os.path.exists('history_all.csv'):
        print("找不到 history_all.csv，請確認資料檔存在。")
        return
        
    df = pd.read_csv('history_all.csv')
    df = preprocess_dataframe(df)

    le = LabelEncoder()
    df['station_idx'] = le.fit_transform(df['ID'])
    num_stations = df['station_idx'].nunique()

    SEQ_LENGTH = 24 
    X_seq_all, X_id_all, y_all, all_scalers = build_dataset(df, le, seq_length=SEQ_LENGTH) 
    
    train_size = int(len(X_seq_all) * 0.8)
    X_seq_train, X_seq_test = X_seq_all[:train_size], X_seq_all[train_size:]
    X_id_train, X_id_test = X_id_all[:train_size], X_id_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    # 輸入特徵數量從 9 變成 11 (4個數值 + 7個時間特徵)
    ts_input = Input(shape=(SEQ_LENGTH, 11), name='ts_input')
    
    # 增加 Conv1D 層以提取局部時間特徵 (CNN-LSTM 架構)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(ts_input)
    x = BatchNormalization()(x)
    
    # 加入 L2 正則化防止過擬合，設定 return_sequences=True 供下層 LSTM 及 Attention 使用
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    
    # 加入 Self-Attention 機制
    attn_out = Attention()([x, x])
    # 利用 GlobalAveragePooling1D 將時間維度壓縮保留全局特徵
    x = GlobalAveragePooling1D()(attn_out)
    x = BatchNormalization()(x)

    id_input = Input(shape=(1,), name='id_input')
    emb = Embedding(input_dim=num_stations, output_dim=16)(id_input)
    emb = Flatten()(emb)

    merged = Concatenate()([x, emb])
    merged = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(32, activation='relu')(merged)
    output = Dense(2, name='out')(merged)

    model = Model(inputs=[ts_input, id_input], outputs=output)
    optimizer = Adam(learning_rate=0.001)
    
    # 改用 Huber Loss 減少突發異常數據對模型訓練的干擾
    model.compile(optimizer=optimizer, loss=Huber())

    early_stop = ProgressEarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    print("開始訓練模型...")
    history = model.fit(
        [X_seq_train, X_id_train], y_train,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("顯示訓練過程可視化圖表 (關閉圖表後繼續)...")
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss (訓練集損失)')
    plt.plot(history.history['val_loss'], label='Val Loss (驗證集損失)')
    plt.title('Model Training Loss (Huber Loss 模型訓練損失)')
    plt.xlabel('Epochs (迭代次數)')
    plt.ylabel('Loss (損失)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.save('bike_model_v2.h5')
    with open('model_assets.pkl', 'wb') as f:
        pickle.dump({'scalers': all_scalers, 'le': le}, f)
    print("模型與相關資源已儲存成功。")

    print("進行模型評估預測...")
    y_pred_scaled = model.predict([X_seq_test, X_id_test])
    show_evaluation_plots(y_test, y_pred_scaled, X_id_test, le, all_scalers)


def evaluate_existing_model():
    print("\n--- 載入現有模型進行評估 ---")
    try:
        model = load_model('bike_model_v2.h5', custom_objects=custom_objects)
        with open('model_assets.pkl', 'rb') as f:
            assets = pickle.load(f)
        all_scalers = assets['scalers'] 
        le = assets['le']
    except Exception as e:
        print(f"載入失敗：{e}。請確認已存在 bike_model_v2.h5 與 model_assets.pkl。")
        return

    if not os.path.exists('history_all.csv'):
        print("找不到 history_all.csv，請確認資料檔存在。")
        return

    df = pd.read_csv('history_all.csv')
    df = preprocess_dataframe(df)
    
    try:
        df['station_idx'] = le.transform(df['ID'])
    except ValueError:
        print("資料集中包含未參與訓練的新站點，請重新訓練模型或清理資料。")
        return

    SEQ_LENGTH = 24
    X_seq_all, X_id_all, y_all, _ = build_dataset(df, le, scalers=all_scalers, seq_length=SEQ_LENGTH)
    
    train_size = int(len(X_seq_all) * 0.8)
    X_seq_test = X_seq_all[train_size:]
    X_id_test = X_id_all[train_size:]
    y_test = y_all[train_size:]

    print("進行模型評估預測...")
    y_pred_scaled = model.predict([X_seq_test, X_id_test])
    show_evaluation_plots(y_test, y_pred_scaled, X_id_test, le, all_scalers)


def interactive_predict():
    print("\n--- 互動式預測 ---")
    try:
        model = load_model('bike_model_v2.h5', custom_objects=custom_objects)
        with open('model_assets.pkl', 'rb') as f:
            assets = pickle.load(f)
        all_scalers = assets['scalers'] 
        le = assets['le']
        if not os.path.exists('history_all.csv'):
            print("找不到 history_all.csv。")
            return
        df = pd.read_csv('history_all.csv')
    except Exception as e:
        print(f"載入失敗：{e}。請先執行選項 1 進行模型訓練。")
        return

    station_names = {
        19005: "臺大水源舍區A棟", 19006: "臺大卓越研究大樓", 19007: "臺大水源修齊會館", 19008: "臺大檔案展示館", 
        19009: "臺大水源舍區B棟", 19043: "臺大男八舍東側", 19044: "臺大禮賢樓東南側", 19045: "臺大農業陳列館北側", 
        19046: "臺大管理學院二館北側", 19047: "臺大土木系館", 19048: "臺大大一女舍北側", 19049: "臺大女九舍西南側", 
        19050: "臺大小福樓東側", 19051: "臺大立體機車停車場", 19052: "臺大工綜館南側", 19053: "臺大天文數學館南側", 
        19054: "臺大心理系館南側", 19055: "臺大樂學館東側", 19056: "臺大農化新館西側", 19057: "臺大五號館西側", 
        19058: "臺大舊體育館西側", 19059: "臺大共同教室北側", 19060: "臺大共同教室東南側", 19061: "臺大鹿鳴堂東側", 
        19062: "臺大公館停車場西北側", 19063: "臺大第二行政大樓南側", 19064: "臺大明達館機車停車場", 19065: "臺大二號館", 
        19066: "臺大凝態館南側", 19067: "臺大社科院西側", 19068: "臺大社會系館南側", 19069: "臺大思亮館東南側", 
        19070: "臺大椰林小舖", 19071: "臺大計資中心南側", 19072: "臺大原分所北側", 19074: "臺大生命科學館西北側", 
        19075: "臺大第一活動中心西南側", 19076: "臺大博理館西側", 19077: "臺大博雅館西側", 19078: "臺大森林館北側", 
        19079: "臺大一號館", 19080: "臺大小小福西南側", 19081: "臺大教研館北側", 19082: "臺大四號館東北側", 
        19083: "臺大新生教室南側", 19084: "臺大鄭江樓北側", 19085: "臺大電機二館東南側", 19086: "臺大圖資系館北側", 
        19087: "臺大總圖書館西南側", 19088: "臺大黑森林西側", 19089: "臺大獸醫館南側", 19090: "臺大新體育館東南側", 
        19091: "臺大明達館北側(員工宿舍)", 19092: "臺大管理學院一館", 19093: "臺大禮賢樓西側", 19094: "臺大大一女餐廳廣場", 
        19095: "臺大學新館", 19096: "臺大水源舍區C棟", 19097: "臺大人文館", 19098: "臺大數學研究中心(東側)",
    }

    def get_station_display_name(sid):
        return f"{station_names.get(sid, '未知站點')} ({sid})"

    print("\n[可用站點清單]:")
    available_ids = sorted(df['ID'].unique())
    for i, sid in enumerate(available_ids):
        print(f"{get_station_display_name(sid):<25}", end="\t" if (i + 1) % 2 != 0 else "\n")
    print()

    try:
        user_input_id = int(input("\n請輸入欲預測的站點 ID (例如 19070): "))
        if user_input_id not in le.classes_:
            print("此 ID 不在訓練資料中")
            return

        print("請輸入欲預測的目標時間 (將預測從最新資料時間點至目標時間的所有變化)")
        user_input_time = input("目標時間 (格式: YYYY-MM-DD HH:00, 範例: 2024-11-01 15:00): ")
        target_dt = datetime.strptime(user_input_time, "%Y-%m-%d %H:%M")

        s_idx = le.transform([user_input_id])[0] 
        scaler = all_scalers[s_idx]
        
        df = preprocess_dataframe(df) # 確保時間序列連續

        station_data = df[df['ID'] == user_input_id].sort_values('time')
        last_row = station_data.iloc[-1]
        current_time = last_row['time']

        if target_dt <= current_time: 
            print(f"時間必須晚於資料最後記錄點 ({current_time})")
            return

        print(f"\n[計算中] 正在分析 {get_station_display_name(user_input_id)} 的預測趨勢...")

        recent_24h = station_data.tail(24).copy()
        scaled_init = scaler.transform(recent_24h[['rent', 'return', 'times used', 'net rent']])

        def get_time_features(dt): 
            return [
                np.sin(2 * np.pi * dt.hour / 24), np.cos(2 * np.pi * dt.hour / 24),
                np.sin(2 * np.pi * dt.weekday() / 7), np.cos(2 * np.pi * dt.weekday() / 7),
                1.0 if dt.weekday() >= 5 else 0.0,            # is_weekend
                1.0 if dt.hour in [7, 8, 9] else 0.0,         # is_morning_peak
                1.0 if dt.hour in [17, 18, 19] else 0.0       # is_evening_peak
            ]

        current_sequence = []
        for i in range(24):
            row_time = current_time - timedelta(hours=(23-i))
            current_sequence.append(list(scaled_init[i]) + get_time_features(row_time))

        current_sequence = np.array(current_sequence)
        temp_time = current_time
        
        predicted_times = []
        predicted_rents = []
        predicted_returns = []

        total_hours = int((target_dt - current_time).total_seconds() // 3600)
        if total_hours <= 0: total_hours = 1
        step = 0

        while temp_time < target_dt:
            step += 1
            print(f"\r預測進度: {step}/{total_hours} 小時 ({(step/total_hours)*100:.1f}%)", end="", flush=True)
            temp_time += timedelta(hours=1)
            
            # shape 改為 11 (4個數值 + 7個時間特徵)
            input_ts = current_sequence.reshape(1, 24, 11) 
            # 使用 model(..., training=False) 替代 model.predict() 以大幅加速單筆迴圈推論的速度
            pred_scaled = model([input_ts, np.array([s_idx])], training=False).numpy()

            # 將預測出來的 rent 和 return 補齊 4 維以進行反標準化
            pred_padded = np.zeros((1, 4))
            pred_padded[0, :2] = pred_scaled[0]
            res = scaler.inverse_transform(pred_padded)
            
            pred_rent = round(max(0, res[0][0]), 2)
            pred_return = round(max(0, res[0][1]), 2)
            pred_times_used = pred_rent + pred_return
            pred_net_rent = pred_rent - pred_return
            
            # 把四個數值一起再標準化，準備做為下一步的輸入
            new_features_raw = np.array([[pred_rent, pred_return, pred_times_used, pred_net_rent]])
            new_features_scaled = scaler.transform(new_features_raw)[0]

            new_row = np.hstack([new_features_scaled, get_time_features(temp_time)])
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            predicted_times.append(temp_time)
            predicted_rents.append(pred_rent)
            predicted_returns.append(pred_return)

        print() # 換行避免覆蓋進度條
        print("\n" + "="*40)
        print(f"站點名稱：{get_station_display_name(user_input_id)}")
        print(f"預測時間段：{current_time + timedelta(hours=1)} 至 {target_dt}")
        print("="*40)
        
        for t, r, ret in zip(predicted_times, predicted_rents, predicted_returns):
            print(f"時間: {t} | 預計租借量: {r} | 預計歸還量: {ret}")

        # 產生預測視覺化圖片
        print("\n顯示預測趨勢可視化圖表 (關閉圖表後繼續)...")
        plt.figure(figsize=(12, 6))
        plt.plot(predicted_times, predicted_rents, label='Predicted Rent (預計租借量)', marker='o', color='blue', alpha=0.7)
        plt.plot(predicted_times, predicted_returns, label='Predicted Return (預計歸還量)', marker='s', color='orange', alpha=0.7)
        plt.title(f'Future Prediction Trend for {get_station_display_name(user_input_id)} (未來預測趨勢)')
        plt.xlabel('Time (時間)')
        plt.ylabel('Amount of Bikes (車輛數)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("輸入格式錯誤，請檢查 ID 與時間格式。")
    except Exception as e:
        print(f"系統錯誤: {e}")


def main():
    # 切換工作目錄至腳本所在目錄，確保能讀取到歷史資料與模型
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    while True:
        print("\n" + "="*40)
        print("      YouBike LSTM 互動與預測系統")
        print("="*40)
        print("1. 訓練 LSTM 模型並顯示訓練損失可視化結果")
        print("2. 載入現有模型並顯示評估可視化圖表 (免重新訓練)")
        print("3. 輸入未來的時間段進行預測 (並顯示趨勢圖表)")
        print("4. 退出")
        choice = input("請選擇操作 (1/2/3/4): ")

        if choice == '1':
            train_model()
        elif choice == '2':
            evaluate_existing_model()
        elif choice == '3':
            interactive_predict()
        elif choice == '4':
            print("退出系統。")
            break
        else:
            print("無效的選擇，請重新輸入。")

if __name__ == "__main__":
    main()