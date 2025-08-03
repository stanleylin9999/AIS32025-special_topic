import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy.fft import fft



#1. 讀取 St Petersburg INCART 12-lead Arrhythmia Database
data_dir = 'C:/python程式/AIS32025專題/'
record_ids = [f"I{str(i).zfill(2)}" for i in range(1, 76)]
X = []
window = 50  # 每段 100 點

for rec_id in record_ids:
    try:
        path = os.path.join(data_dir, rec_id)
        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')
        signal = record.p_signal[:, 0]  # 只取第1導程

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

#3. 模擬中間人攻擊
def simulate_mitm_attack(ecg_signal, noise_level=0.5, mode='', replacement_pool=None):
    signal = np.array(ecg_signal).copy().reshape(100, 1)
    signal = (signal - np.mean(signal)) / np.std(signal)

    if mode == 'add_noise':
        noise = np.random.normal(0.0, noise_level, size=signal.shape)
        signal += noise
    
    else:
        raise ValueError(f"未知的中間人攻擊模式：{mode}")

    return signal.reshape(-1)

#提取特徵
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft
import numpy as np

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


def get_attack_features(X_attack, mode, replacement_pool):
    tampered_samples = [
        simulate_mitm_attack(sample, noise_level=0.8, mode=mode, replacement_pool=replacement_pool)
        for sample in X_attack
    ]
    tampered_samples = np.array(tampered_samples)
    tampered_feat = np.array([extract_features(seg) for seg in tampered_samples])
    return tampered_feat

attack_modes = ['add_noise']


index = 0  # 可以改成其他 index
original_signal = X_attack[index]

# 模擬中間人攻擊
attacked_signal = simulate_mitm_attack(original_signal, noise_level=0.8, mode='add_noise')

# 畫圖比較
plt.figure(figsize=(12, 5))
plt.plot(original_signal, label='Original ECG', linewidth=2)
plt.plot(attacked_signal, label='Tampered ECG (add_noise)', linestyle='--')
plt.title('ECG Signal Before and After MITM Attack')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
