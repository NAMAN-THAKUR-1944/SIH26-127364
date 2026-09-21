"""Logistic Regression Baseline Model.

Implements a static classifier (Logistic Regression) over the flattened state
vectors S_t. This serves as the lower-bound baseline before we implement
the Latent Dynamics (LSTM/Transformer) World Model.
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

LOGGER = logging.getLogger(__name__)

class BaselineLR:
    """Logistic Regression baseline for temporal state vectors S_t."""
    
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=random_state, max_iter=max_iter, class_weight="balanced"))
        ])
        self.feature_names = []
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract purely numeric features for training/prediction."""
        # Drop metadata and label columns
        exclude = {"window_start", "window_end", "is_attack", "attack_labels", "active_flow_count"}
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        # Return a copy to avoid SettingWithCopyWarning if we pad later
        return df[features].copy().astype(float)
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """Train the logistic regression model."""
        if train_df.empty:
            raise ValueError("Training dataframe is empty")
            
        X = self._extract_features(train_df)
        y = train_df["is_attack"].fillna(0).astype(int)
        
        self.feature_names = list(X.columns)
        
        # In case there's only one class in training
        if len(np.unique(y)) < 2:
            LOGGER.warning("Training set has only one class! Model might not generalize.")
            
        self.pipeline.fit(X, y)
        LOGGER.info("Trained BaselineLR on %d samples with %d features.", len(X), len(self.feature_names))
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict infiltration probabilities."""
        if df.empty:
            return np.array([])
            
        X = self._extract_features(df)
        
        # Ensure feature alignment
        missing = set(self.feature_names) - set(X.columns)
        for col in missing:
            X[col] = 0.0
            
        # Reorder to match training
        X = X[self.feature_names]
            
        return self.pipeline.predict_proba(X)[:, 1]
        
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary infiltration status."""
        prob = self.predict_proba(df)
        if len(prob) == 0:
            return np.array([])
        return (prob >= threshold).astype(int)
        
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model and return Precision, Recall, F1, and AUROC."""
        if test_df.empty:
            return {}
            
        y_true = test_df["is_attack"].fillna(0).astype(int)
        if len(np.unique(y_true)) < 2:
            LOGGER.warning("Test set has only one class; AUROC may be undefined.")
            
        y_pred = self.predict(test_df)
        y_prob = self.predict_proba(test_df)
        
        metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auroc"] = np.nan
            
        return metrics

def _selftest() -> None:
    # Create synthetic data
    train_data = pd.DataFrame([
        {"window_start": 0, "byte_count_sum": 100, "is_attack": 0},
        {"window_start": 10, "byte_count_sum": 120, "is_attack": 0},
        {"window_start": 20, "byte_count_sum": 5000, "is_attack": 1},
        {"window_start": 30, "byte_count_sum": 6000, "is_attack": 1},
    ])
    
    test_data = pd.DataFrame([
        {"window_start": 40, "byte_count_sum": 110, "is_attack": 0},
        {"window_start": 50, "byte_count_sum": 5500, "is_attack": 1},
    ])
    
    model = BaselineLR()
    model.fit(train_data)
    
    metrics = model.evaluate(test_data)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auroc"] == 1.0
    
    print("baseline_lr selftest: OK")
    
if __name__ == "__main__":
    _selftest()
