# 學生成績分析 App

## 專案簡介
本專案為一套以 Streamlit 為基礎的學生成績分析應用，支援 Excel 成績檔案上傳、互動式資料篩選、多元統計分析與視覺化，並結合 AI（Gemini API）產生個人化學習建議。

## 主要功能特色
- 首頁 Dashboard：快速總覽學生人數、平均分數、及格率與分數分布
- 分頁式分析：分數分布、通過率、趨勢、分群、預測、相關性等多元分析
- 互動式圖表：支援 Plotly 互動直方圖、箱型圖、折線圖，hover 顯示細節、可縮放
- Sidebar 篩選：學校、年級、檢測名稱、分群等多層次篩選，支援全選/全不選
- AI 建議：各分析分頁可呼叫 Gemini API 產生指標說明與個人化建議
- 程式碼模組化，易於維護與擴充

## 安裝與執行方式
1. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動應用程式：
   ```bash
   streamlit run app.py
   ```
3. 於網頁介面上傳 Excel 成績檔案（.xlsx），即可開始分析

## 主要模組說明
- `app.py`：主程式，負責分頁調度、資料流轉、AI 建議整合
- `modules/sidebar.py`：側邊欄篩選區，支援多層次篩選與分群
- `modules/dashboard.py`：首頁 Dashboard，顯示總覽指標與分布
- `modules/score_distribution.py`：分數分布分析，含靜態與互動圖表
- `modules/score_clustering.py`：成績分群分析（KMeans），分群摘要與分布
- `modules/score_trend.py`：成績分布與趨勢分析
- `modules/time_series.py`：時間序列分析，觀察多次檢測/分組趨勢
- `modules/auto_test_clustering.py`：自動分群檢測分布分析
- 其餘模組：通過率、預測、相關性、個案追蹤、交叉分析等

## 優化紀錄與維護建議
- 2024/06/10~13：完成 UI/UX 架構重整、分析功能強化、AI 建議、互動圖表、程式碼模組化與註解優化
- 詳細優化歷程請見《優化計畫.md》
- 建議每次功能優化後，於《優化計畫.md》與本 README 同步紀錄

---
如需串接真實 Gemini API，請於 `get_gemini_advice` 相關函式補上 API Key 與串接邏輯。

如有新需求或問題，請於 issues 或專案討論區提出。

## Gemini API 串接說明

- 本專案於 `app.py` 中的 `get_gemini_advice` 函式負責呼叫 Gemini API 產生分析建議。
- 預設會讀取環境變數 `GEMINI_API_KEY` 作為 API 金鑰，若未設定則自動 fallback 為內建 mock 建議（不會實際呼叫 API）。
- 若要啟用真實 Gemini 服務，請於系統環境變數設定：
  ```bash
  export GEMINI_API_KEY=你的API金鑰
  ```
- 可於 `get_gemini_advice` 內自訂 prompt 結構，根據 context、stats、group、context_info 等參數產生不同分析建議。
- 若 API 呼叫失敗，會自動顯示錯誤訊息並 fallback 為 mock 建議。
- 相關串接邏輯可參考 `app.py` 內 `get_gemini_advice` 與 `mock_gemini_advice` 實作。 