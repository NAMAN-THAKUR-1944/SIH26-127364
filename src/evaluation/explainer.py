"""Explainability module using PyTorch Gradients.

Extracts the driving original input features that led to a specific K-step
infiltration prediction, mapping gradients back through the Autoencoder.
Due to Windows DLL restrictions with Numba/SHAP, this uses pure PyTorch 
Input-X-Gradient attribution which serves the exact same purpose.
"""

import logging
from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np

from src.models.world_model import CyberWorldModel

LOGGER = logging.getLogger(__name__)

class WorldModelWrapper(nn.Module):
    """Wraps the world model for gradient compatibility."""
    def __init__(self, model: CyberWorldModel, k_steps: int):
        super().__init__()
        self.model = model
        self.k_steps = k_steps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, future_probs = self.model.simulate(x, k_steps=self.k_steps, return_logits=True)
        # return mean probability across the K steps (batch, 1)
        return future_probs.mean(dim=1)

class InfiltrationExplainer:
    def __init__(self, model: CyberWorldModel, k_steps: int = 5):
        """
        Args:
            model: Trained CyberWorldModel
            k_steps: The prediction horizon being explained.
        """
        self.model = model
        self.model.eval()
        self.k_steps = k_steps
        self.wrapper = WorldModelWrapper(model, k_steps)
        
    def explain(self, context_x: torch.Tensor, feature_names: List[str]) -> Dict[str, float]:
        """Explain a specific context sequence's prediction using Input*Gradient.
        
        Args:
            context_x: (1, seq_len, input_dim) tensor
            feature_names: List of feature names of length input_dim
            
        Returns:
            Dictionary mapping feature name to its total absolute importance.
        """
        # Require gradients for the input tensor
        context_x = context_x.clone().detach().requires_grad_(True)
        
        # Forward pass
        pred = self.wrapper(context_x)
        
        # Backward pass to get gradients w.r.t input
        pred.backward(torch.ones_like(pred))
        
        gradients = context_x.grad
        
        # Input * Gradient attribution
        attribution = (gradients * context_x).detach().cpu().numpy()
        
        # Aggregate importance across the sequence length (mean of absolute values)
        # shape: (1, seq_len, input_dim)
        attr_abs = np.abs(attribution).mean(axis=1).squeeze() # Shape: (input_dim,)
        
        importance_map = {}
        for i, name in enumerate(feature_names):
            importance_map[name] = float(attr_abs[i])
            
        # Sort descending
        importance_map = dict(sorted(importance_map.items(), key=lambda item: item[1], reverse=True))
        return importance_map

def _selftest():
    import warnings
    warnings.filterwarnings("ignore")
    
    input_dim = 5
    seq_len = 3
    k_steps = 2
    
    model = CyberWorldModel(input_dim=input_dim, latent_dim=8, hidden_dim=16)
    
    explainer = InfiltrationExplainer(model, k_steps=k_steps)
    
    # Test data
    x = torch.randn(1, seq_len, input_dim)
    feat_names = [f"feat_{i}" for i in range(input_dim)]
    
    imp = explainer.explain(x, feat_names)
    
    assert len(imp) == input_dim
    assert list(imp.keys())[0] in feat_names
    
    print("explainer selftest: OK")

if __name__ == "__main__":
    _selftest()
