ECG MITM Attack Detection using PCA & Machine Learning

本專案針對心電圖訊號在傳輸過程中遭受中間人攻擊的場景，設計了一套基於多維度特徵擷取、PCA 降維與Logistic Regression 的異常偵測機制。

專案同時採用了兩個公開心電圖資料集進行驗證：MIT-BIH Arrhythmia Database 與 St Petersburg INCART 12-lead Arrhythmia Database，兩者皆包含 100 維的訊號特徵，但由於檔案格式與讀檔方式不同，因此分開提供兩版對應的程式碼與測試結果。


專題檔案結構


.
├── README.md                                          # 專題說明文件
├── pca(MIT-BIH Arrhythmia Database).py                # MIT-BIH 資料集處理與訓練主程式
├── pca(St Petersburg INCART 12-lead Arrhythmia...).py  # St Petersburg INCART 資料集處理與訓練主程式
├── mit_pca.png                                        # MIT-BIH 的 3D PCA 降維與決策平面視覺化圖
├── mit_混淆矩陣.png                                    # MIT-BIH 的混淆矩陣
├── st_pca.png                                         # St Petersburg INCART 的 3D PCA 視覺化圖
└── st_混淆矩陣.png                                     # St Petersburg INCART 的混淆矩陣
