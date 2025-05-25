import numpy as np
import neurokit2 as nk
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")

class EEGProcessor:
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate

    def calculate_features(self, signal: np.ndarray, rr_intervals: np.ndarray) -> Dict[str, float]:
        features = {}
        
        peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=self.sampling_rate)
        
        hrv_time = nk.hrv_time(peaks, sampling_rate=self.sampling_rate)
        features['Mean_RR'] = hrv_time['HRV_MeanNN'].iloc[0] if 'HRV_MeanNN' in hrv_time else 0
        features['SDNN'] = hrv_time['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in hrv_time else 0
        features['RMSSD'] = hrv_time['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in hrv_time else 0
        features['pNN50'] = hrv_time['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in hrv_time else 0
        
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=self.sampling_rate)
        features['LF_power'] = hrv_freq['HRV_LF'].iloc[0] if 'HRV_LF' in hrv_freq else 0
        features['HF_power'] = hrv_freq['HRV_HF'].iloc[0] if 'HRV_HF' in hrv_freq else 0
        
        if features['HF_power'] > 0:
            features['LF_HF_ratio'] = features['LF_power'] / features['HF_power']
        else:
            features['LF_HF_ratio'] = 0
            
        if features['Mean_RR'] > 0:
            features['Mean_HR'] = 60 / features['Mean_RR']
        else:
            features['Mean_HR'] = 0
        
        return features

    def segment_signal(self, signal: np.ndarray, segment_length: int = 10, overlap: float = 0.5) -> List[np.ndarray]:
        """Разбиение сигнала на сегменты с перекрытием."""
        step = int(segment_length * self.sampling_rate * (1 - overlap))
        segments = []
        
        for i in range(0, len(signal) - segment_length * self.sampling_rate, step):
            segment = signal[i:i + segment_length * self.sampling_rate]
            segments.append(segment)
            
        return segments
