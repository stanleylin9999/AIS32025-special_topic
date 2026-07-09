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




data_dir = 'C:/python程式/AIS32025專題/'
record_ids = [str(i) for i in range(100, 125) if i not in [110, 120]]
record_ids += [str(i) for i in range(200, 235) if i not in [204, 206, 211, 216, 218, 224, 225, 226, 227, 228, 229]]

X = []
window = 50

for rec_id in record_ids:
    
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


X = np.array(X)



X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
X_train, X_temp = train_test_split(X_norm, test_size=0.4, random_state=42)
X_val, X_attack = train_test_split(X_temp, test_size=0.5, random_state=42)

def simulate_mitm_attack(ecg_signal, noise_level=0.5, mode='', replacement_pool=None):
    signal = np.array(ecg_signal).copy().reshape(100, 1)
    signal = (signal - np.mean(signal)) / np.std(signal)

    if mode == 'add_noise':
        noise = np.random.normal(0.0, noise_level, size=signal.shape)
        signal += noise

    return signal.reshape(-1)



def extract_features(segment):
    segment = np.array(segment)
    diff = np.diff(segment)
    
    
    fft_vals = np.abs(fft(segment))[:5]

    
    peaks, _ = find_peaks(segment)
    valleys, _ = find_peaks(-segment)

    
    peak_heights = segment[peaks] if len(peaks) > 0 else np.array([0])
    valley_heights = segment[valleys] if len(valleys) > 0 else np.array([0])
    peak_max_height = np.max(peak_heights) if len(peak_heights) > 0 else 0

    
    rms = np.sqrt(np.mean(segment**2))
    
    
    psd = np.abs(fft(segment))**2
    psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
    spectral_entropy = entropy(psd_norm)

    
    peak_intervals = np.diff(peaks) if len(peaks) > 1 else np.array([0])

    features = [
        np.mean(segment),                   
        np.std(segment),                    
        np.max(segment),                    
        np.min(segment),                    
        np.median(segment),                 
        np.sum(np.abs(segment)),            
        np.sum(segment ** 2),               
        np.mean(diff),                     
        np.std(diff),                       
        np.sum(np.diff(np.sign(segment)) != 0),  
        skew(segment),                     
        kurtosis(segment),                 
        np.max(diff),                      
        np.min(diff),                      
        len(peaks),                        
        len(valleys),                      
        np.mean(peak_heights),             
        peak_max_height,                   
        np.mean(valley_heights),           
        np.mean(peak_intervals),           
        rms,                               
        spectral_entropy,                  
        *fft_vals                          
    ]

    return np.array(features)



X_train_feat = np.array([extract_features(seg) for seg in X_train])
pca = PCA(n_components=3)
pca.fit(X_train_feat)
train_pca = pca.transform(X_train_feat)


def get_attack_features(X_attack, mode, replacement_pool):
    tampered_samples = [
        simulate_mitm_attack(sample, noise_level=0.8, mode=mode, replacement_pool=replacement_pool)
        for sample in X_attack
    ]
    tampered_samples = np.array(tampered_samples)
    tampered_feat = np.array([extract_features(seg) for seg in tampered_samples])
    return tampered_feat


attack_modes = ['add_noise']

for mode in attack_modes:
    attack_feat = get_attack_features(X_attack, mode, X_norm)
    attack_pca = pca.transform(attack_feat)
    
    X_cls = np.vstack([train_pca, attack_pca])
    y_cls = np.hstack([np.zeros(len(train_pca)), np.ones(len(attack_pca))])
    
    clf = LogisticRegression()
    clf.fit(X_cls, y_cls)
    
    
    w = clf.coef_[0]
    b = clf.intercept_[0]

    
    xx, yy = np.meshgrid(
        np.linspace(X_cls[:, 0].min(), X_cls[:, 0].max(), 20),
        np.linspace(X_cls[:, 1].min(), X_cls[:, 1].max(), 20)
    )
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]

    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], c='green', label='Training', s=15, alpha=0.4)
    ax.scatter(attack_pca[:, 0], attack_pca[:, 1], attack_pca[:, 2], c='red', label=f'Attack: {mode}', s=20, alpha=1)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue', label='Decision Plane')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(f'3D PCA Projection - Mode: {mode}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    y_pred = clf.predict(X_cls)
    y_true = y_cls

    
    cm = confusion_matrix(y_true, y_pred)

    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='YlGnBu', values_format='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
