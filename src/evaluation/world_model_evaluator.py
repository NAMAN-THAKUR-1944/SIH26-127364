"""Evaluation framework for the Latent Dynamics World Model.

Takes the simulated K-step future state predictions from the CyberWorldModel
and evaluates them against the true temporal attack progression sequences.
Computes metrics (Precision, Recall, F1, AUROC) over moving averages to
validate MITRE ground truth progression.
"""

import logging
from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.models.world_model import CyberWorldModel

LOGGER = logging.getLogger(__name__)

class WorldModelEvaluator:
    def __init__(self, model: CyberWorldModel, k_steps: int = 5, threshold: float = 0.5):
        self.model = model
        self.k_steps = k_steps
        self.threshold = threshold
        # Put model in eval mode
        self.model.eval()

    def evaluate_sequence(self, context_x: torch.Tensor, future_y_true: np.ndarray, future_labels: List[str]) -> Dict[str, float]:
        """Evaluate a single contextual window against its true K-step future.
        
        Args:
            context_x: (1, seq_len, input_dim) tensor
            future_y_true: (k_steps,) binary array of true attacks
            future_labels: List of length k_steps containing MITRE attack labels
            
        Returns:
            Dictionary of metrics for this sequence
        """
        with torch.no_grad():
            _, future_probs = self.model.simulate(context_x, k_steps=self.k_steps)
            
        # future_probs shape: (1, k_steps, 1) -> flatten to (k_steps,)
        y_prob = future_probs.squeeze().cpu().numpy()
        
        # Moving average (in this case, just the mean of the K-step predictions vs the true presence)
        # But for sequence prediction, we evaluate per-step or aggregated.
        # Let's calculate per-step metrics and an overall sequence metric.
        
        if len(y_prob.shape) == 0:
            y_prob = np.array([y_prob])
            
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # If the whole future window is safe or whole future window is attack
        try:
            auroc = float(roc_auc_score(future_y_true, y_prob)) if len(np.unique(future_y_true)) > 1 else np.nan
        except Exception:
            auroc = np.nan
            
        return {
            "mean_pred_prob": float(y_prob.mean()),
            "true_attack_ratio": float(future_y_true.mean()),
            "auroc": auroc,
            "y_prob": y_prob.tolist(),
            "y_pred": y_pred.tolist(),
            "y_true": future_y_true.tolist(),
            "labels": future_labels
        }

    def evaluate_dataset(self, df: pd.DataFrame, context_len: int = 10, feature_cols: List[str] = None) -> Dict[str, float]:
        """Sliding window evaluation over the entire dataset."""
        if df.empty or len(df) < context_len + self.k_steps:
            LOGGER.warning("Dataset too small for evaluation.")
            return {}
            
        if feature_cols is None:
            exclude = {"window_start", "window_end", "is_attack", "attack_labels", "active_flow_count"}
            feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
            
        # Convert to arrays for fast sliding
        X_data = df[feature_cols].fillna(0.0).values
        Y_data = df["is_attack"].fillna(0).astype(int).values
        
        # Safe extraction for labels
        if "attack_labels" in df.columns:
            L_data = df["attack_labels"].astype(str).values
        else:
            L_data = np.array(["BENIGN"] * len(df))
            
        all_y_true = []
        all_y_prob = []
        
        for i in range(len(df) - context_len - self.k_steps + 1):
            # Extract context
            ctx = X_data[i : i + context_len]
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0) # (1, seq_len, dim)
            
            # Extract future ground truth
            future_idx = i + context_len
            fut_y = Y_data[future_idx : future_idx + self.k_steps]
            
            with torch.no_grad():
                _, future_probs = self.model.simulate(ctx_tensor, k_steps=self.k_steps)
                
            prob = future_probs.squeeze().cpu().numpy()
            if len(prob.shape) == 0:
                prob = np.array([prob])
                
            all_y_true.extend(fut_y)
            all_y_prob.extend(prob)
            
        y_true_flat = np.array(all_y_true)
        y_prob_flat = np.array(all_y_prob)
        y_pred_flat = (y_prob_flat >= self.threshold).astype(int)
        
        metrics = {
            "precision": float(precision_score(y_true_flat, y_pred_flat, zero_division=0)),
            "recall": float(recall_score(y_true_flat, y_pred_flat, zero_division=0)),
            "f1": float(f1_score(y_true_flat, y_pred_flat, zero_division=0)),
        }
        
        try:
            metrics["auroc"] = float(roc_auc_score(y_true_flat, y_prob_flat))
        except ValueError:
            metrics["auroc"] = np.nan
            
        LOGGER.info(f"World Model Evaluation [K={self.k_steps}]: F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}")
        return metrics

def _selftest():
    from src.models.world_model import CyberWorldModel
    
    # Synthetic dataset
    np.random.seed(42)
    torch.manual_seed(42)
    
    context_len = 5
    k_steps = 3
    input_dim = 4
    
    df = pd.DataFrame({
        "f1": np.random.randn(20),
        "f2": np.random.randn(20),
        "f3": np.random.randn(20),
        "f4": np.random.randn(20),
        "is_attack": np.random.randint(0, 2, 20),
        "attack_labels": ["BENIGN"] * 10 + ["Botnet"] * 10
    })
    
    model = CyberWorldModel(input_dim=input_dim, latent_dim=8, hidden_dim=16)
    evaluator = WorldModelEvaluator(model, k_steps=k_steps)
    
    metrics = evaluator.evaluate_dataset(df, context_len=context_len, feature_cols=["f1", "f2", "f3", "f4"])
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    print("world_model_evaluator selftest: OK")

if __name__ == "__main__":
    _selftest()
