import numpy as np
import neurokit2 as nk
from typing import List, Dict
import warnings
import logging


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")


class EEGProcessor:
    """
    Класс для обработки и анализа ЭКГ сигналов.

    Attributes:
        sampling_rate (int): Частота дискретизации сигнала в Гц
    """

    def __init__(self, sampling_rate: int = 1000) -> None:
        """
        Инициализирует процессор ЭКГ сигналов.

        Args:
            sampling_rate (int, optional):
                Частота дискретизации в Гц. default=1000.
        """
        self.sampling_rate = sampling_rate
        logger.info("EEGProcessor initialized with " +
                    f"sampling rate {sampling_rate} Hz")

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Извлекает признаки из ЭКГ сигнала.

        Args:
            signal (np.ndarray): Массив значений ЭКГ сигнала

        Returns:
            np.ndarray: Массив извлеченных признаков

        Raises:
            ValueError: Если сигнал пустой или некорректный
        """
        logger.info("Extracting features from signal with " +
                    f"shape {signal.shape}")

        segments = self.segment_signal(signal, overlap=0.9)

        features_list = []
        for i, segment in enumerate(segments):
            logger.debug(f"Processing segment {i+1}/{len(segments)}")

            r_peaks = nk.ecg_findpeaks(segment,
                                       sampling_rate=self.sampling_rate)
            rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / self.sampling_rate

            if len(rr_intervals) > 0:
                features = self.calculate_features(segment, rr_intervals)
                feature_array = np.array([features[feature] for feature in [
                    'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50',
                    'LF_power', 'HF_power', 'LF_HF_ratio'
                ]])
                features_list.append(feature_array)

        if not features_list:
            logger.warning("No features were extracted from the signal")
            raise ValueError("No features could be extracted from the signal")

        logger.info("Successfully extracted " +
                    f"{len(features_list)} feature sets")
        return np.array(features_list)

    def calculate_features(self, signal: np.ndarray,
                           rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Рассчитывает признаки из ЭКГ сигнала и RR-интервалов.

        Args:
            signal (np.ndarray): Массив значений ЭКГ сигнала
            rr_intervals (np.ndarray): Массив RR-интервалов в секундах

        Returns:
            Dict[str, float]: Словарь с рассчитанными признаками:
                - Mean_RR: Средний RR-интервал
                - SDNN: Стандартное отклонение RR-интервалов
                - RMSSD: Корень из среднего квадрата разностей
                    последовательных RR-интервалов
                - pNN50: Процент RR-интервалов, отличающихся более чем на 50 мс
                - LF_power: Мощность в низкочастотном диапазоне
                - HF_power: Мощность в высокочастотном диапазоне
                - LF_HF_ratio: Отношение LF/HF
                - Mean_HR: Средняя частота сердечных сокращений
        """
        features = {}

        try:
            peaks = nk.intervals_to_peaks(rr_intervals,
                                          sampling_rate=self.sampling_rate)

            hrv_time = nk.hrv_time(peaks, sampling_rate=self.sampling_rate)
            features['Mean_RR'] = hrv_time.get('HRV_MeanNN', 0).iloc[0]
            features['SDNN'] = hrv_time.get('HRV_SDNN', 0).iloc[0]
            features['RMSSD'] = hrv_time.get('HRV_RMSSD', 0).iloc[0]
            features['pNN50'] = hrv_time.get('HRV_pNN50', 0).iloc[0]

            hrv_freq = nk.hrv_frequency(peaks,
                                        sampling_rate=self.sampling_rate)
            features['LF_power'] = hrv_freq.get('HRV_LF', 0).iloc[0]
            features['HF_power'] = hrv_freq.get('HRV_HF', 0).iloc[0]

            if features['HF_power'] > 0:
                features['LF_HF_ratio'] = (
                    features['LF_power'] / features['HF_power']
                )
            else:
                features['LF_HF_ratio'] = 0

            if features['Mean_RR'] > 0:
                features['Mean_HR'] = 60 / features['Mean_RR']
            else:
                features['Mean_HR'] = 0

            return features

        except Exception as e:
            logger.error("Error calculating features: " +
                         f"{str(e)}", exc_info=True)
            return {
                'Mean_RR': 0, 'SDNN': 0, 'RMSSD': 0, 'pNN50': 0,
                'LF_power': 0, 'HF_power': 0, 'LF_HF_ratio': 0, 'Mean_HR': 0
            }

    def segment_signal(self, signal: np.ndarray,
                       segment_length: int = 10,
                       overlap: float = 0.5) -> List[np.ndarray]:
        """
        Разбивает сигнал на сегменты с перекрытием.

        Args:
            signal (np.ndarray): Массив значений ЭКГ сигнала
            segment_length (int, optional):
                Длина сегмента в секундах. default=10.
            overlap (float, optional):
                Доля перекрытия между сегментами (от 0 до 1). default=0.5.

        Returns:
            List[np.ndarray]: Список сегментов сигнала

        Raises:
            ValueError: Если overlap не в диапазоне [0, 1) или
                segment_length <= 0
        """
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in range [0, 1)")
        if segment_length <= 0:
            raise ValueError("Segment length must be positive")

        logger.info("Segmenting signal with " +
                    f"length {len(signal)} into {segment_length}s segments " +
                    f"with {overlap*100}% overlap")

        step = int(segment_length * self.sampling_rate * (1 - overlap))
        segments = []

        for i in range(0, len(signal) - segment_length * self.sampling_rate,
                       step):
            segment = signal[i:i + segment_length * self.sampling_rate]
            segments.append(segment)

        logger.info(f"Created {len(segments)} segments")
        return segments
