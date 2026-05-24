import pandas as pd
import os
import itertools

'''
資料預處理

'''
# 下載 U-bike 2.0的即時資訊 轉成df

file_url = 'https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json' # 若非雲端執行要改路徑 or 下載後放在同一資料夾
stationInfo = pd.read_json(file_url) #dataframe
stationInfo.info()

# 定義需要欄位
toRemove = [
    'sareaen', 'snaen', 'aren', 'act', 'srcUpdateTime', 
    'updateTime', 'infoTime', 'infoDate', 'mday', 'ar'
]
stationInfo.drop(toRemove, axis=1, inplace=True, errors='ignore')
stationInfo.sno = stationInfo.sno % 100000 #簡化id
stationInfo.sna = stationInfo.sna.str[11:] #取地區

# 4. 重新命名：把長長的名字改成好用的短名字
stationInfo.rename({
    'sno': 'ID', 
    'sna': 'name', 
    'available_rent_bikes': 'sbi',      
    'available_return_bikes': 'bemp'    
}, axis=1, inplace=True)

# 5. 篩選台大公館校區
stationInfo = stationInfo[stationInfo['sarea'] == '臺大公館校區']
stationInfo.drop('sarea', axis=1, inplace=True)
stationInfo.reset_index(drop=True, inplace=True)
print(stationInfo.head())


stationInfo.to_csv('stationInfo.csv') #站名csv檔

'''
資料對齊
'''
stationID = stationInfo[['ID', 'name']]
colName = ['rentTime', 'rentStation', 'returnTime', 'returnStation', 'rent', 'infoDate']

#特徵工程
folder = '/kaggle/input/taipei-youbike-2-0-rental-records/unzipped'
csv_path = lambda yymm: os.path.join(folder, f'20{yymm}.csv')

#count 
def load_df(df: pd.DataFrame, rtype:str):
    _station = rtype + 'Station'; _time = rtype + 'Time'
    df = df.groupby(_station)[_time].value_counts().reset_index()
    df = df.merge(stationID, 'right', left_on=_station, right_on='name') #把站名對齊，合併後會有ID欄位
    df.dropna(inplace=True) #刪除空值
    df.drop(columns=[_station, 'name'], inplace=True)
    df = df.rename(columns={'count': rtype})
    df[_time] = pd.to_datetime(df[_time])
    #轉換成 年/月/日/小時
    df2 = df[['ID', rtype]].assign(
        year= df[_time].dt.year,
        month= df[_time].dt.month,
        day= df[_time].dt.day,
        hour= df[_time].dt.hour
    )
    return df2

#統整22-24年份的data

merge_on = ['ID', 'year', 'month', 'day', 'hour']
column_order = merge_on + ['rent', 'return']
history_all = pd.DataFrame(columns=column_order).astype(int) #ID ,時間 ,借 ,還

# 增加資料量：擴充涵蓋 2022、2023、2024 全年份資料 (若資料集存在)
for yymm in itertools.chain(range(2201, 2213), range(2301, 2313), range(2401, 2413)):
    print(yymm)
    df = pd.read_csv(csv_path(yymm), names=colName, encoding_errors='replace')
    df = pd.merge(
        load_df(df, 'rent'), load_df(df, 'return'),
        on=merge_on, how='outer' #外連接，確保若偵測到沒人還車,補0
    ).fillna(0).astype(int)
    history_all = pd.concat(
        [history_all, df],
        ignore_index=True
    )

#補上細項欄位 ( 日期、星期幾、淨租借數量 )
history_all['times used'] = history_all['rent'] + history_all['return']
history_all['net rent'] = history_all['rent'] - history_all['return']
history_all['day_of_week'] = pd.to_datetime(history_all[['year', 'month', 'day']]).dt.dayofweek
history_all.head()


history_all.to_csv('history_all.csv', index=False) #最終DATA