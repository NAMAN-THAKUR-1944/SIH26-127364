"""Latent Dynamics World Model Architecture.

Implements the core PyTorch neural networks for the Cyber Defense World Model.
- StateEncoder: Compresses flattened state vectors S_t into latent space z_t.
- StateDecoder: Reconstructs S_t from z_t (for self-supervised training).
- LatentDynamics: LSTM-based recurrent model predicting z_{t+1} from z_t.
- InfiltrationClassifier: Predicts the probability of attack from z_t.
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional

LOGGER = logging.getLogger(__name__)

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class StateDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        """Uses an LSTM to model the temporal evolution of the latent state."""
        super().__init__()
        # We use an LSTM to capture sequential dynamics robustly.
        # It takes z_t and previous hidden state (h_{t-1}, c_{t-1}) and predicts z_{t+1}
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.z_predictor = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z_seq: torch.Tensor, hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            z_seq: Tensor of shape (batch, seq_len, latent_dim)
            hc: Optional hidden states
        Returns:
            predicted_z: (batch, seq_len, latent_dim)
            new_hc: Tuple of updated hidden states
        """
        lstm_out, hc = self.lstm(z_seq, hc)
        z_pred = self.z_predictor(lstm_out)
        return z_pred, hc

class InfiltrationClassifier(nn.Module):
    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class CyberWorldModel(nn.Module):
    """The unified Cyber Defense World Model."""
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = StateEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = StateDecoder(latent_dim, hidden_dim, input_dim, dropout)
        self.dynamics = LatentDynamics(latent_dim, hidden_dim)
        self.classifier = InfiltrationClassifier(latent_dim, dropout)
        
    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass on a sequence.
        
        Args:
            x_seq: (batch, seq_len, input_dim)
        Returns:
            reconstructed_x: (batch, seq_len, input_dim)
            predicted_next_z: (batch, seq_len, latent_dim) -> predicted z_{t+1}
            attack_probs: (batch, seq_len, 1) -> predicted from z_t
            z_seq: (batch, seq_len, latent_dim) -> true encoded latents
        """
        batch_size, seq_len, input_dim = x_seq.shape
        
        # 1. Encode all states
        # Flatten batch and seq_len for linear layer
        x_flat = x_seq.view(-1, input_dim)
        z_flat = self.encoder(x_flat)
        z_seq = z_flat.view(batch_size, seq_len, -1)
        
        # 2. Decode for reconstruction loss
        x_recon_flat = self.decoder(z_flat)
        x_recon = x_recon_flat.view(batch_size, seq_len, input_dim)
        
        # 3. Predict next latents
        # Predict z_{t+1} from z_{1...t}
        z_next_pred, _ = self.dynamics(z_seq)
        
        # 4. Classify current states
        attack_logits_flat = self.classifier(z_flat)
        attack_probs_flat = torch.sigmoid(attack_logits_flat)
        attack_probs = attack_probs_flat.view(batch_size, seq_len, 1)
        
        return x_recon, z_next_pred, attack_probs, z_seq

    def compute_loss(self, x_seq: torch.Tensor, y_seq: torch.Tensor, loss_weights: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        """Compute the joint loss for training the World Model.
        
        Args:
            x_seq: (batch, seq_len, input_dim)
            y_seq: (batch, seq_len, 1) true labels
            loss_weights: Optional weights for the different loss components.
            
        Returns:
            total_loss: scalar tensor
            metrics: dict of loss components
        """
        if loss_weights is None:
            loss_weights = {"recon": 1.0, "dynamics": 1.0, "cls": 1.0}
            
        x_recon, z_next_pred, attack_probs, z_seq = self.forward(x_seq)
        
        # 1. Reconstruction Loss: ensure z represents the current state
        recon_loss = nn.functional.mse_loss(x_recon, x_seq)
        
        # 2. Dynamics Loss: ensure LSTM predicts the actual next latent state
        # Compare prediction at t with actual latent at t+1
        if x_seq.shape[1] > 1:
            dynamics_loss = nn.functional.mse_loss(z_next_pred[:, :-1, :], z_seq[:, 1:, :].detach())
        else:
            dynamics_loss = torch.tensor(0.0, device=x_seq.device)
            
        # 3. Classification Loss: ensure z contains threat semantics
        cls_loss = nn.functional.binary_cross_entropy(attack_probs, y_seq)
        
        total_loss = (loss_weights["recon"] * recon_loss +
                      loss_weights["dynamics"] * dynamics_loss +
                      loss_weights["cls"] * cls_loss)
                      
        metrics = {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_dynamics": dynamics_loss.item(),
            "loss_cls": cls_loss.item()
        }
        return total_loss, metrics

    def simulate(self, x_start: torch.Tensor, k_steps: int, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unroll the world model K steps into the future.
        
        Args:
            x_start: (batch, seq_len, input_dim) - Initial context sequence.
            k_steps: Number of future steps to simulate.
            return_logits: If True, returns raw classification logits instead of probabilities.
            
        Returns:
            future_z: (batch, k_steps, latent_dim)
            future_attack_probs: (batch, k_steps, 1)
        """
        batch_size, seq_len, input_dim = x_start.shape
        
        # Encode initial context
        x_flat = x_start.view(-1, input_dim)
        z_context = self.encoder(x_flat).view(batch_size, seq_len, -1)
        
        # Get predictions and hidden state from context
        z_pred, hc = self.dynamics(z_context)
        
        # The prediction for T+1 is the last element of z_pred
        next_z = z_pred[:, -1:, :]
        
        future_z_list = [next_z]
        
        logits = self.classifier(next_z)
        prob = logits if return_logits else torch.sigmoid(logits)
        future_probs_list = [prob]
        
        current_z = next_z
        
        # Autoregressive K-step prediction (for the remaining K-1 steps)
        for _ in range(k_steps - 1):
            next_z, hc = self.dynamics(current_z, hc)
            future_z_list.append(next_z)
            
            logits = self.classifier(next_z)
            prob = logits if return_logits else torch.sigmoid(logits)
            future_probs_list.append(prob)
            
            current_z = next_z
            
        future_z = torch.cat(future_z_list, dim=1)
        future_probs = torch.cat(future_probs_list, dim=1)
        
        return future_z, future_probs

def _selftest():
    batch_size = 4
    seq_len = 10
    input_dim = 20
    k_steps = 5
    
    model = CyberWorldModel(input_dim=input_dim, latent_dim=16, hidden_dim=32, dropout=0.1)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    
    # Test forward
    x_recon, z_next_pred, attack_probs, z_seq = model(x)
    assert x_recon.shape == (batch_size, seq_len, input_dim)
    assert z_next_pred.shape == (batch_size, seq_len, 16)
    assert attack_probs.shape == (batch_size, seq_len, 1)
    assert z_seq.shape == (batch_size, seq_len, 16)
    
    # Test compute_loss
    loss, metrics = model.compute_loss(x, y)
    assert isinstance(loss, torch.Tensor)
    assert "loss_total" in metrics
    
    # Test simulate
    future_z, future_probs = model.simulate(x, k_steps=k_steps)
    assert future_z.shape == (batch_size, k_steps, 16)
    assert future_probs.shape == (batch_size, k_steps, 1)
    
    print("world_model selftest: OK")

if __name__ == "__main__":
    _selftest()
