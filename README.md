ECG MITM Attack Detection using PCA & Machine Learning

本專案針對心電圖訊號在傳輸過程中遭受中間人攻擊的場景，設計了一套基於多維度特徵擷取、PCA 降維與Logistic Regression 的異常偵測機制。

專案同時採用了兩個公開心電圖資料集進行驗證：MIT-BIH Arrhythmia Database 與 St Petersburg INCART 12-lead Arrhythmia Database，兩者皆包含 100 維的訊號特徵，但由於檔案格式與讀檔方式不同，因此分開提供兩版對應的程式碼與測試結果。

