import numpy as np
from pathlib import Path
import mne
import neurokit2 as nk
from typing import Dict, List, Tuple
from model_manager import ModelManager
import warnings
import json

from features.eeg_processor import EEGProcessor


warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="neurokit2.signal.signal_psd")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="neurokit2.hrv.hrv_time")


def load_ecg(file_path: str, sampling_rate: int = 1000) -> np.ndarray:
    """Загрузка ЭКГ сигнала из .edf файла."""
    raw = mne.io.read_raw_edf(file_path, preload=True)
    ecg_channel = next((ch for ch in raw.ch_names
                        if 'ECG' in ch.upper() or 'EKG' in ch.upper()), None)
    if ecg_channel is None:
        raise ValueError("Канал ЭКГ не найден в файле.")
    return raw.get_data(picks=ecg_channel)[0]


def process_file(file_path: str,
                 intervals: Dict[str, List[Tuple[float, float]]],
                 sampling_rate: int = 1000) -> Dict[str, List[np.ndarray]]:
    """Обработка одного файла и извлечение признаков для каждого состояния."""
    processor = EEGProcessor(sampling_rate=sampling_rate)
    signal = load_ecg(file_path, sampling_rate)

    features_by_state = {
        "rest": [],
        "load": [],
        "recovery": []
    }

    for state, segments in intervals.items():
        for start_time, end_time in segments:
            start_idx = int(start_time * sampling_rate)
            end_idx = int(end_time * sampling_rate)
            segment_signal = signal[start_idx:end_idx]

            sub_segments = processor.segment_signal(segment_signal,
                                                    overlap=0.9,
                                                    segment_length=10)

            for sub_segment in sub_segments:
                r_peaks = nk.ecg_findpeaks(sub_segment,
                                           sampling_rate=sampling_rate)
                rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate

                if len(rr_intervals) > 0:
                    features = processor.calculate_features(sub_segment,
                                                            rr_intervals)
                    feature_names = [
                        'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50',
                        'LF_power', 'HF_power', 'LF_HF_ratio'
                    ]
                    feature_array = np.array([
                        features[feature] for feature in feature_names
                    ])
                    features_by_state[state].append(feature_array)

    return features_by_state


def load_data(markup_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка и обработка данных с разметкой"""
    all_features = []
    all_labels = []

    if not Path(markup_path).exists():
        raise FileNotFoundError(f"Файл разметки {markup_path} не найден")

    with open(markup_path, 'r') as f:
        data_markup = json.load(f)

    for file_path, intervals in data_markup.items():
        print(f"Обработка файла: {file_path}")
        features_by_state = process_file(file_path, intervals)

        for state, features_list in features_by_state.items():
            if features_list:
                all_features.extend(features_list)
                all_labels.extend([state] * len(features_list))

    return np.array(all_features), np.array(all_labels)


def main():
    markup_path = Path("data_markup.json")
    model_manager = ModelManager()
    X, y = load_data(markup_path)

    feature_names = [
        'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50',
        'LF_power', 'HF_power', 'LF_HF_ratio'
    ]

    model, metrics = model_manager.train_model(X, y, feature_names)

    print("Модель обучена и сохранена")
    print(f"Метрики: {metrics}")


if __name__ == "__main__":
    main()
