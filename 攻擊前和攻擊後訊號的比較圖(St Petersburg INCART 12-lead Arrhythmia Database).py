import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
from scipy.signal import find_peaks




data_dir = 'C:/python程式/AIS32025專題/'
record_ids = [f"I{str(i).zfill(2)}" for i in range(1, 76)]
X = []
window = 50 

for rec_id in record_ids:
    
        path = os.path.join(data_dir, rec_id)
        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')
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


def get_attack_features(X_attack, mode, replacement_pool):
    tampered_samples = [
        simulate_mitm_attack(sample, noise_level=0.8, mode=mode, replacement_pool=replacement_pool)
        for sample in X_attack
    ]
    tampered_samples = np.array(tampered_samples)
    tampered_feat = np.array([extract_features(seg) for seg in tampered_samples])
    return tampered_feat

attack_modes = ['add_noise']


index = 0
original_signal = X_attack[index]


attacked_signal = simulate_mitm_attack(original_signal, noise_level=0.8, mode='add_noise')


plt.figure(figsize=(12, 5))
plt.plot(original_signal, label='Original ECG', linewidth=2)
plt.plot(attacked_signal, label='After add noise ECG', linestyle='--')
plt.title('ECG Signal Before and After MITM Attack')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
