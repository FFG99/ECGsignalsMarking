import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pickle
import optuna
import neurokit2 as nk
import logging

from features.eeg_processor import EEGProcessor


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Класс для управления моделью для классификации ЭКГ сигналов.

    Attributes:
        model_dir (Path): Директория для сохранения модели и связанных файлов
        scaler (StandardScaler): Объект для нормализации признаков
        label_encoder (LabelEncoder): Объект для кодирования меток классов
    """

    def __init__(self) -> None:
        """
        Инициализирует ModelManager и создает необходимые директории.
        """
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        logger.info("ModelManager initialized")

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения модели.

        Args:
            X (np.ndarray): Матрица признаков
            y (np.ndarray): Вектор меток классов

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        logger.info(f"Preparing data with shapes: X={X.shape}, y={y.shape}")

        y = y.ravel()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Data prepared" +
                    f"train={X_train.shape}, test={X_test.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X: np.ndarray, y: np.ndarray,
                    feature_names: List[str]) -> Tuple[CatBoostClassifier,
                                                       Dict[str, float]]:
        """
        Обучает модель CatBoost с оптимизацией гиперпараметров.

        Args:
            X (np.ndarray): Матрица признаков
            y (np.ndarray): Вектор меток классов
            feature_names (List[str]): Список имен признаков

        Returns:
            Tuple[CatBoostClassifier, Dict[str, float]]:
                (обученная модель, метрики качества)
        """
        logger.info("Starting model training")

        X_train_scaled, X_test_scaled, y_train, y_test = (
            self.prepare_data(X, y)
        )

        def objective(trial: optuna.Trial) -> float:
            """
            Целевая функция для оптимизации гиперпараметров.

            Args:
                trial (optuna.Trial): Объект для оптимизации

            Returns:
                float: Значение метрики качества
            """
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': (
                    trial.suggest_float('learning_rate', 0.01, 0.3)
                ),
                'depth': trial.suggest_int('depth', 3, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': 'Bernoulli',
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_seed': 42,
                'verbose': 100,
                'early_stopping_rounds': 50,
                'eval_metric': 'MultiClass',
                'loss_function': 'MultiClass',
                'classes_count': len(np.unique(y))
            }

            model = CatBoostClassifier(**params)
            eval_set = [(X_test_scaled, y_test)]

            model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                use_best_model=True
            )

            y_pred = model.predict(X_test_scaled)
            return accuracy_score(y_test, y_pred)

        logger.info("Starting hyperparameter optimization")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        best_params.update({
            'bootstrap_type': 'Bernoulli',
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'classes_count': len(np.unique(y))
        })

        logger.info(f"Best parameters found: {best_params}")

        model = CatBoostClassifier(**best_params)
        eval_set = [(X_test_scaled, y_test)]

        logger.info("Training final model with best parameters")
        model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            use_best_model=True
        )

        y_pred = model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)

        logger.info(f"Model training completed with metrics: {metrics}")
        self.save_model(model, feature_names, metrics)

        return model, metrics

    def _calculate_metrics(self, y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Рассчитывает метрики качества модели.

        Args:
            y_true (np.ndarray): Истинные метки классов
            y_pred (np.ndarray): Предсказанные метки классов

        Returns:
            Dict[str, float]: Словарь с метриками качества
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }

        report = classification_report(y_true, y_pred, output_dict=True)
        for label in report:
            if isinstance(report[label], dict):
                for metric, value in report[label].items():
                    metrics[f'{label}_{metric}'] = value

        return metrics

    def save_model(self, model: CatBoostClassifier,
                   feature_names: List[str],
                   metrics: Dict[str, float]) -> None:
        """
        Сохраняет модель и связанные файлы.

        Args:
            model (CatBoostClassifier): Обученная модель
            feature_names (List[str]): Список имен признаков
            metrics (Dict[str, float]): Метрики качества модели
        """
        logger.info("Saving model and related files")

        model.save_model(str(self.model_dir / 'model.cbm'))

        with open(self.model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(self.model_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        metadata = {
            'feature_names': feature_names,
            'metrics': metrics
        }
        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info("Model and related files saved successfully")

    @staticmethod
    def load_model() -> Tuple[CatBoostClassifier, StandardScaler,
                              LabelEncoder, Dict]:
        """
        Загружает сохраненную модель и связанные файлы.

        Returns:
            Tuple[CatBoostClassifier, StandardScaler, LabelEncoder, Dict]:
                (модель, scaler, label_encoder, метаданные)

        Raises:
            FileNotFoundError: Если модель не найдена
        """
        logger.info("Loading model and related files")

        model_dir = Path("model")

        if not model_dir.exists():
            logger.error("Model directory not found")
            raise FileNotFoundError("Модель не найдена")

        model = CatBoostClassifier()
        model.load_model(str(model_dir / 'model.cbm'))

        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        logger.info("Model and related files loaded successfully")
        return model, scaler, label_encoder, metadata

    @staticmethod
    def predict(model: CatBoostClassifier, scaler: StandardScaler,
                label_encoder: LabelEncoder, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания с помощью модели.

        Args:
            model (CatBoostClassifier): Обученная модель
            scaler (StandardScaler): Объект для нормализации признаков
            label_encoder (LabelEncoder): Объект для декодирования меток
            X (np.ndarray): Матрица признаков

        Returns:
            np.ndarray: Предсказанные метки классов
        """
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return label_encoder.inverse_transform(predictions)

    @staticmethod
    def predict_record(model: CatBoostClassifier, scaler: StandardScaler,
                       label_encoder: LabelEncoder, record: np.ndarray,
                       sampling_rate: int = 1000,
                       window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Делает предсказания для записи ЭКГ с объединением соседних интервалов.

        Args:
            model (CatBoostClassifier): Обученная модель
            scaler (StandardScaler): Объект для нормализации признаков
            label_encoder (LabelEncoder): Объект для декодирования меток
            record (np.ndarray): Запись ЭКГ
            sampling_rate (int, optional): Частота дискретизации. default=1000.
            window_size (int, optional): Размер окна в секундах. default=10.

        Returns:
            List[Dict[str, Any]]: Список предсказаний с временными метками
        """
        logger.info(f"Processing record with length {len(record)}")

        processor = EEGProcessor(sampling_rate=sampling_rate)
        segments = processor.segment_signal(record, segment_length=window_size,
                                            overlap=0.9)

        predictions = []
        for i, segment in enumerate(segments):
            logger.debug(f"Processing segment {i+1}/{len(segments)}")

            r_peaks = nk.ecg_findpeaks(segment, sampling_rate=sampling_rate)
            rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate

            if len(rr_intervals) > 0:
                features = processor.calculate_features(segment, rr_intervals)
                feature_array = np.array([features[feature] for feature in [
                    'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50',
                    'LF_power', 'HF_power', 'LF_HF_ratio'
                ]])

                prediction = ModelManager.predict(model, scaler, label_encoder,
                                                  feature_array.reshape(
                                                      1, -1)
                                                  )[0]
                start_time = i * window_size * (1 - 0.9)  # overlap
                end_time = start_time + window_size
                predictions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'prediction': prediction
                })

        # Объединяем соседние интервалы с одинаковыми предсказаниями
        merged_predictions = []
        if predictions:
            current_pred = predictions[0]

            for next_pred in predictions[1:]:
                if (next_pred['prediction'] == current_pred['prediction'] and
                    next_pred['start_time'] - current_pred['end_time'] <
                        window_size * 0.1):  # small gap
                    current_pred['end_time'] = next_pred['end_time']
                else:
                    merged_predictions.append(current_pred)
                    current_pred = next_pred

            merged_predictions.append(current_pred)

        logger.info(f"Generated {len(merged_predictions)} merged predictions")
        return merged_predictions
