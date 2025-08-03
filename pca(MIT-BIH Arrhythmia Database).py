import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft
import numpy as np


#1.讀取MIT-BIH Arrhythmia Database

data_dir = 'C:/python程式/AIS32025專題/'
record_ids = [str(i) for i in range(100, 125) if i not in [110, 120]]
record_ids += [str(i) for i in range(200, 235) if i not in [204, 206, 211, 216, 218, 224, 225, 226, 227, 228, 229]]

X = []
window = 50

for rec_id in record_ids:
    try:
        record_path = data_dir + rec_id
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal[:, 0]
        

        for idx in annotation.sample:
            if idx < window or idx + window >= len(signal):
                continue
            segment = signal[idx - window: idx + window]
            if len(segment) == 100:
                X.append(segment)

        print(f'完成讀取：{rec_id}')
    except Exception as e:
        print(f'讀取 {rec_id} 時出錯：{e}')

X = np.array(X)


# 2. 標準化並切分訓練/驗證/攻擊資料
X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
X_train, X_temp = train_test_split(X_norm, test_size=0.4, random_state=42)
X_val, X_attack = train_test_split(X_temp, test_size=0.5, random_state=42)

def simulate_mitm_attack(ecg_signal, noise_level=0.5, mode='', replacement_pool=None):
    signal = np.array(ecg_signal).copy().reshape(100, 1)
    signal = (signal - np.mean(signal)) / np.std(signal)

    if mode == 'add_noise':
        noise = np.random.normal(0.0, noise_level, size=signal.shape)
        signal += noise
    
    else:
        raise ValueError(f"未知的中間人攻擊模式：{mode}")

    return signal.reshape(-1)

#3. 提取特徵

def extract_features(segment):
    segment = np.array(segment)
    diff = np.diff(segment)
    
    # 頻域特徵：FFT
    fft_vals = np.abs(fft(segment))[:5]

    # 找波峰與波谷
    peaks, _ = find_peaks(segment)
    valleys, _ = find_peaks(-segment)

    # 波峰/谷高度
    peak_heights = segment[peaks] if len(peaks) > 0 else np.array([0])
    valley_heights = segment[valleys] if len(valleys) > 0 else np.array([0])
    peak_max_height = np.max(peak_heights) if len(peak_heights) > 0 else 0

    # RMS
    rms = np.sqrt(np.mean(segment**2))
    
    # 頻譜熵
    psd = np.abs(fft(segment))**2
    psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
    spectral_entropy = entropy(psd_norm)

    # 波峰間距離均值
    peak_intervals = np.diff(peaks) if len(peaks) > 1 else np.array([0])

    features = [
        np.mean(segment),                   # 平均值
        np.std(segment),                    # 標準差
        np.max(segment),                    # 最大值
        np.min(segment),                    # 最小值
        np.median(segment),                 # 中位數
        np.sum(np.abs(segment)),            # 絕對振幅總和
        np.sum(segment ** 2),               # 能量（平方和）
        np.mean(diff),                      # 一階差分平均
        np.std(diff),                       # 一階差分標準差
        np.sum(np.diff(np.sign(segment)) != 0),  # 零交叉數
        skew(segment),                     # 偏態
        kurtosis(segment),                 # 峰度
        np.max(diff),                      # 最大斜率
        np.min(diff),                      # 最小斜率
        len(peaks),                        # 波峰數
        len(valleys),                      # 波谷數
        np.mean(peak_heights),             # 波峰高度平均
        peak_max_height,                   # 波峰最大值（新特徵）
        np.mean(valley_heights),           # 波谷高度平均
        np.mean(peak_intervals),           # 波峰間距平均
        rms,                               # RMS
        spectral_entropy,                  # 頻譜熵
        *fft_vals                          # 前5個 FFT 特徵
    ]

    return np.array(features)


# 4.用特徵向量擬合 PCA
X_train_feat = np.array([extract_features(seg) for seg in X_train])
pca = PCA(n_components=3)
pca.fit(X_train_feat)
train_pca = pca.transform(X_train_feat)

# 5.攻擊資料特徵
def get_attack_features(X_attack, mode, replacement_pool):
    tampered_samples = [
        simulate_mitm_attack(sample, noise_level=0.8, mode=mode, replacement_pool=replacement_pool)
        for sample in X_attack
    ]
    tampered_samples = np.array(tampered_samples)
    tampered_feat = np.array([extract_features(seg) for seg in tampered_samples])
    return tampered_feat

# 6.畫出3D圖
attack_modes = ['add_noise']

for mode in attack_modes:
    attack_feat = get_attack_features(X_attack, mode, X_norm)
    attack_pca = pca.transform(attack_feat)
    # 訓練分類器（攻擊樣本標記為 1，正常為 0）
    X_cls = np.vstack([train_pca, attack_pca])
    y_cls = np.hstack([np.zeros(len(train_pca)), np.ones(len(attack_pca))])
    
    clf = LogisticRegression()
    clf.fit(X_cls, y_cls)
    
    # 取得分類器權重
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # 計算平面：w1*x + w2*y + w3*z + b = 0
    # 解出 z = (-w1*x - w2*y - b) / w3
    xx, yy = np.meshgrid(
        np.linspace(X_cls[:, 0].min(), X_cls[:, 0].max(), 20),
        np.linspace(X_cls[:, 1].min(), X_cls[:, 1].max(), 20)
    )
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    # 畫圖
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], c='green', label='Training', s=15, alpha=0.4)
    ax.scatter(attack_pca[:, 0], attack_pca[:, 1], attack_pca[:, 2], c='red', label=f'Attack: {mode}', s=20, alpha=1)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue', label='Decision Plane')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(f'3D PCA Projection + Separation Plane - Mode: {mode}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # 1. 做預測
    y_pred = clf.predict(X_cls)
    y_true = y_cls

    # 2. 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)

    # 3. 顯示混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='YlGnBu', values_format='d', ax=ax)
    plt.title(f'Confusion Matrix - Tampered ECG ({mode} Attack)')
    plt.tight_layout()
    plt.show()
