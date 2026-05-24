from flask import Flask, render_template, request, jsonify, Response
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
import tensorflow as tf
import os

# 確保在同一工作目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# 解決 Keras 載入帶有自訂 Loss 的模型問題
custom_objects = {"mse": tf.keras.losses.MeanSquaredError(), "Huber": Huber()}

print("正在初始化 Web 應用程式並載入模型與資料...")
try:
    model = load_model('bike_model_v2.h5', custom_objects=custom_objects)
    with open('model_assets.pkl', 'rb') as f:
        assets = pickle.load(f)
    all_scalers = assets['scalers']
    le = assets['le']
    
    # 從 Interactive_Ubike_App 匯入前處理函數以維持一致的特徵處理
    from Interactive_Ubike_App import preprocess_dataframe
    raw_df = pd.read_csv('history_all.csv')
    df = preprocess_dataframe(raw_df)
    print("Web 應用程式初始化完成！")
except Exception as e:
    print(f"初始化失敗: {e}。請確保已執行 python Interactive_Ubike_App.py 並完成了一次模型訓練 (選項1)。")
    df = None

# YouBike 站點名稱對照表
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

@app.route('/')
def index():
    if df is None:
        return "資料或模型載入失敗，請先確保 Interactive_Ubike_App.py 已經成功訓練過一次模型。"
    
    # 建立讓前端選單使用的資料
    stations = [{"id": int(sid), "name": station_names.get(sid, f"未知站點 ({sid})")} for sid in sorted(df['ID'].unique())]
    return render_template('index.html', stations=stations)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    user_input_id = int(data.get('station_id'))
    start_dt_str = data.get('start_time')
    target_dt_str = data.get('target_time')
    
    if not start_dt_str or not target_dt_str:
         return jsonify({"error": "必須提供起始與結束時間"}), 400

    try:
        start_dt = datetime.strptime(start_dt_str, "%Y-%m-%dT%H:%M")
        target_dt = datetime.strptime(target_dt_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return jsonify({"error": "時間格式錯誤"}), 400

    if target_dt <= start_dt:
        return jsonify({"error": "結束預測時間必須晚於起始預測時間"}), 400

    if user_input_id not in le.classes_:
        return jsonify({"error": "此 ID 不在訓練資料中"}), 400

    s_idx = le.transform([user_input_id])[0] 
    scaler = all_scalers[s_idx]

    station_data = df[df['ID'] == user_input_id].sort_values('time')
    db_last_time = station_data.iloc[-1]['time']
    if isinstance(db_last_time, str):
        db_last_time = pd.to_datetime(db_last_time)

    # 決定模型推論的起點時間
    if start_dt <= db_last_time:
        past_data = station_data[station_data['time'] <= start_dt]
        if len(past_data) < 24:
            return jsonify({"error": "起始時間之前的歷史資料不足 24 小時，無法作為預測起點"}), 400
        recent_24h = past_data.tail(24).copy()
        current_time = past_data.iloc[-1]['time']
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

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
            row_time = recent_24h.iloc[i]['time']
            if isinstance(row_time, str):
                row_time = pd.to_datetime(row_time)
            current_sequence.append(list(scaled_init[i]) + get_time_features(row_time))

    else:
        # 【時間跳躍機制】如果起始時間遠大於資料庫最後時間
        # 為了避免模型逐小時推論幾年(非常慢且誤差極大)，我們直接把時間「快轉」
        recent_24h = station_data.tail(24).copy()
        current_time = start_dt

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
        # 我們將最後 24 小時的「借還車狀態」保留，但將「時間特徵」強行替換為目標起始時間前 24 小時的特徵
        for i in range(24):
            row_time = current_time - timedelta(hours=(24-i))
            current_sequence.append(list(scaled_init[i]) + get_time_features(row_time))

    current_sequence = np.array(current_sequence)
    
    total_hours = int((target_dt - current_time).total_seconds() // 3600)
    if total_hours <= 0: total_hours = 1

    def generate():
        nonlocal current_time, current_sequence
        temp_time = current_time
        results = []
        step = 0

        # 迴圈逐小時預測未來
        while temp_time < target_dt:
            step += 1
            temp_time += timedelta(hours=1)
            input_ts = current_sequence.reshape(1, 24, 11) 
            # 使用 model(..., training=False) 替代 model.predict() 以大幅加速單筆迴圈推論的速度
            pred_scaled = model([input_ts, np.array([s_idx])], training=False).numpy()

            # 反標準化
            pred_padded = pd.DataFrame(np.zeros((1, 4)), columns=['rent', 'return', 'times used', 'net rent'])
            pred_padded.iloc[0, :2] = pred_scaled[0]
            res = scaler.inverse_transform(pred_padded)
            
            pred_rent = round(max(0, res[0][0]), 2)
            pred_return = round(max(0, res[0][1]), 2)
            pred_times_used = pred_rent + pred_return
            pred_net_rent = pred_rent - pred_return
            
            # 標準化新的特徵準備餵給下一輪
            new_features_raw = pd.DataFrame([[pred_rent, pred_return, pred_times_used, pred_net_rent]], columns=['rent', 'return', 'times used', 'net rent'])
            new_features_scaled = scaler.transform(new_features_raw)[0]

            new_row = np.hstack([new_features_scaled, get_time_features(temp_time)])
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # 只有當預測時間 >= 起始時間，才將其加入最終結果中
            if temp_time >= start_dt:
                results.append({
                    "time": temp_time.strftime("%Y-%m-%d %H:%M"),
                    "rent": pred_rent,
                    "return": pred_return
                })
            
            # 每預測完一小時，就發送一次進度更新給前端
            progress = int((step / total_hours) * 100)
            yield f"data: {json.dumps({'progress': progress})}\n\n"

        # 最後將完整的預測結果送出
        final_data = {
            "station_name": station_names.get(user_input_id, f"未知站點 ({user_input_id})"),
            "predictions": results
        }
        yield f"data: {json.dumps({'result': final_data})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # 如果使用者沒有安裝 Flask，會在這之前就報錯，若順利執行則啟動伺服器
    print("\n" + "="*50)
    print("啟動 Web UI 伺服器...")
    print("請開啟瀏覽器並前往： http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)