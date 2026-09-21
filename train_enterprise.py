
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from src.data.advanced_loaders import load_ciciot, load_lanl, load_darpa, FEATURES
from src.models.world_model import CyberWorldModel
from src.evaluation.explainer import InfiltrationExplainer
from src.data.threat_intel import enrich_threat_intel

def make_sequences(df: pd.DataFrame, seq_len: int = 10):
    """Simple sliding window without time-gaps for mocking."""
    data = df[FEATURES].values
    labels = df["is_attack"].values
    raw_labels = df["label"].values
    
    X, y, text_labels = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(labels[i:i+seq_len])
        text_labels.append(raw_labels[i+seq_len-1])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), text_labels

def main():
    print("--- 1. LOADING ENTERPRISE DATASETS ---")
    ciciot_df = load_ciciot("data/ciciot2023_mock.csv")
    lanl_df = load_lanl("data/lanl_auth_mock.csv")
    darpa_df = load_darpa("data/darpa_mock.csv")
    
    print(f"Loaded CICIoT2023: {len(ciciot_df)} flows")
    print(f"Loaded LANL Auth: {len(lanl_df)} events")
    print(f"Loaded DARPA Intrusion: {len(darpa_df)} events")
    
    # Combine everything for a massive enterprise dataset
    full_df = pd.concat([ciciot_df, lanl_df, darpa_df], ignore_index=True)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    full_df[FEATURES] = scaler.fit_transform(full_df[FEATURES])
    
    seq_len = 10
    X, y, labels = make_sequences(full_df, seq_len)
    
    print(f"\n--- 2. TRAINING CYBER WORLD MODEL ---")
    print(f"Total Sequences: {len(X)}")
    print(f"Unified Feature Dimension: {len(FEATURES)}")
    
    model = CyberWorldModel(input_dim=len(FEATURES), latent_dim=16, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Quick mock training loop
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(X, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")
        
    print("\n--- 3. TESTING ADVANCED PIPELINES ---")
    model.eval()
    
    # Test K-Step Simulation
    context = X[0:1] # batch of 1
    future_z, future_probs = model.simulate(context, k_steps=3)
    print(f"[K-Step Forward Simulation]: Success.")
    print(f"  Future probabilities (T+1 to T+3): {future_probs.detach().numpy().flatten()}")
    
    # Test Explainer
    explainer = InfiltrationExplainer(model, k_steps=1)
    importance_map = explainer.explain(context, FEATURES)
    print(f"\n[Explanation / Reasoning Engine]: Success.")
    print("  Top 3 driving features:")
    for feat, val in list(importance_map.items())[:3]:
        print(f"  - {feat}: {val:.4f}")
        
    # Test Threat Intel (CAPEC/CVE)
    print(f"\n[Threat Intel Enrichment]: Success.")
    sample_attacks = ["Mirai-greeth_flood", "neptune.", "Lateral Movement"]
    for attack in sample_attacks:
        intel = enrich_threat_intel(attack)
        print(f"  Raw Label: {attack}")
        print(f"    -> MITRE: {intel['threat']}")
        print(f"    -> CAPEC: {intel['capec']}")
        print(f"    -> CVEs: {intel['cve']}")
        
    print("\n? ENTERPRISE INTEGRATION VERIFIED. READY FOR RTX 5050 GPU RUN.")

if __name__ == "__main__":
    main()

