import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
import json
from typing import Dict, List, Tuple
from pathlib import Path
import pickle
import optuna

class ModelManager:
    def __init__(self):
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = y.ravel()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[CatBoostClassifier, Dict[str, float]]:
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(X, y)
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
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
        
        model = CatBoostClassifier(**best_params)
        eval_set = [(X_test_scaled, y_test)]
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            use_best_model=True
        )
        
        y_pred = model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.save_model(model, feature_names, metrics)
        
        return model, metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        report = classification_report(y_true, y_pred, output_dict=True)
        for label in report:
            if isinstance(report[label], dict):
                for metric, value in report[label].items():
                    metrics[f'{label}_{metric}'] = value
        
        return metrics
    
    def save_model(self, model: CatBoostClassifier, feature_names: List[str], metrics: Dict[str, float]):
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
    
    @staticmethod
    def load_model() -> Tuple[CatBoostClassifier, StandardScaler, LabelEncoder, Dict]:
        model_dir = Path("model")
        
        if not model_dir.exists():
            raise FileNotFoundError("Модель не найдена. Сначала обучите модель.")
        
        model = CatBoostClassifier()
        model.load_model(str(model_dir / 'model.cbm'))
        
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, label_encoder, metadata
    
    @staticmethod
    def predict(model: CatBoostClassifier, scaler: StandardScaler, 
               label_encoder: LabelEncoder, X: np.ndarray) -> np.ndarray:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return label_encoder.inverse_transform(predictions)
