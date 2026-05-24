# YouBike LSTM 站點借還預測系統

本專案是一個基於深度學習 LSTM (Long Short-Term Memory) 模型的 YouBike 2.0 借還車數量預測系統。主要針對**台灣大學公館校區**周邊的 YouBike 站點，透過歷史租借紀錄與時間特徵，預測未來各站點的「借出」與「歸還」車輛數。專案內含完整的資料爬取與前處理、模型訓練流程，以及一個視覺化的 Flask Web 應用程式供使用者進行互動式預測。

## 📁 專案檔案結構

*   **`Ubike_data.py`**：資料收集與預處理腳本。負責從台北市政府開放資料 API 抓取即時站點資訊，並結合歷史租借 CSV 檔（預設路徑指向 Kaggle），進行資料對齊與基礎特徵工程，最終輸出給模型使用的 `history_all.csv`。
*   **`Interactive_Ubike_App.py`**：核心模型訓練程式。負責讀取 `history_all.csv`，進行進階時間序列處理（包含無人借還時段的補零、極端值截斷），建立多輸入的 LSTM 模型（結合數值特徵與時間特徵），進行模型訓練並提供訓練結果的可視化評估。訓練完成後會匯出 `bike_model_v2.h5` 與 `model_assets.pkl`。
*   **`web_app.py`**：Flask Web 伺服器程式。載入已訓練好的模型與特徵處理器，提供 API 讓前端呼叫，並使用 Server-Sent Events (SSE) 技術即時回傳逐小時推論的計算進度與最終預測結果。
*   **`templates/index.html`**：Web 應用程式的前端介面。使用 Bootstrap 5 進行排版，並透過 Chart.js 繪製未來借還車數量的預測折線圖。
*   **`environment.yml`**：Conda 環境設定檔，列出了執行本專案所需的所有 Python 套件依賴。
*   **`history_all.csv`**：（需由 `Ubike_data.py` 生成或手動準備）模型訓練用的核心歷史資料集。

## 🛠️ 環境設定與安裝

本專案使用 `conda` 管理虛擬環境。請確保您的系統已安裝 Anaconda 或 Miniconda。

1. **建立 Conda 虛擬環境：**
   開啟終端機（或 Anaconda Prompt），在專案根目錄下執行以下指令：
   ```bash
   conda env create -f environment.yml
   ```
2. **啟動虛擬環境：**
   ```bash
   conda activate ubike_env
   ```

*(備註：`environment.yml` 包含了 `python=3.10`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow` 等核心套件。)*

## 🚀 使用步驟

### 步驟一：資料準備
確保專案目錄下有 `history_all.csv` 資料檔。若需要重新生成，請執行 `Ubike_data.py`。
*(注意：`Ubike_data.py` 中預設讀取歷史紀錄的路徑為 `/kaggle/input/...`，若要在本地端執行，需先下載台北市 YouBike 歷史資料並修改腳本中的 `folder` 路徑。)*

### 步驟二：模型訓練
在啟動 Web UI 之前，**必須至少完成一次模型訓練**以產生模型權重與相關資產。
```bash
python Interactive_Ubike_App.py
```
執行後程式會進行資料前處理、序列建立，並開始訓練 LSTM 模型。訓練完成後會自動產生 `bike_model_v2.h5` 以及 `model_assets.pkl`。

### 步驟三：啟動 Web UI 進行預測
確認模型檔案皆已產生後，啟動 Flask 伺服器：
```bash
python web_app.py
```
啟動成功後，開啟瀏覽器並前往 `http://127.0.0.1:5000`。
在網頁介面中，您可以：
1. 從下拉選單選擇目標 YouBike 站點。
2. 設定「起始預測時間」與「結束預測時間」。
3. 點擊「開始計算」，系統將會在背景逐小時推論，並透過進度條顯示運算狀態。
4. 運算完成後，下方會自動繪製出該站點未來借出與歸還數量的趨勢預測圖。

## 🧠 技術細節

*   **CNN-LSTM 混合架構**：模型不僅使用了 LSTM 捕捉長期時間依賴，還在前端加入了 `Conv1D` 卷積層與 `BatchNormalization`，組成 CNN-LSTM 架構，以更有效地提取局部的時間特徵（例如短時間內的突發借還潮）。
*   **資料集擴充**：預設的資料蒐集範圍從原本的 2023-2024 年，進一步擴展涵蓋 2022 年至 2024 年（共三年）的歷史紀錄，大幅增加訓練樣本數量，提升模型的泛化能力。
*   **資料補全機制**：針對 YouBike 站點在深夜或冷門時段沒有借還紀錄的狀況，程式會自動以小時為單位將時間序列補齊，並填補為 0，確保 LSTM 模型的序列連續性。
*   **極端值截斷 (Outlier Clipping)**：為避免突發的大型活動導致異常極端值影響資料標準化 (MinMaxScaler) 的尺度，程式會取 99 百分位數作為上限進行截斷。
*   **時間特徵工程**：除了基本的年月日時，模型還加入了平滑處理的時間特徵（Hour Sin/Cos, Day Sin/Cos）以及二元狀態特徵（是否為週末、是否為早晚尖峰時段）。
*   **注意力機制 (Attention)**：引入 Self-Attention 機制，讓模型能自動聚焦在 24 小時輸入序列中最重要的時間節點。
*   **自迴歸推論機制**：在 Web 預測階段，針對未來的時間，程式會將前一小時的預測結果當作下一小時的輸入，以此進行滾動式 (Rolling) 的自迴歸預測。
