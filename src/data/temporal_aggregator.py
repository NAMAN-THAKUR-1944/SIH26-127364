"""Temporal Aggregation for SIH Cyber Defense World Model.

Groups aligned flow telemetry into fixed time-windows (Delta t) to generate
a sequence of flattened state vectors (S_t) and their corresponding labels.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

def aggregate_windows(
    aligned_df: pd.DataFrame,
    window_size_s: float = 10.0,
    stride_s: Optional[float] = None
) -> pd.DataFrame:
    """Group flows into time windows and aggregate features.
    
    Args:
        aligned_df: DataFrame from telemetry_aligner.
        window_size_s: Size of the time window in seconds.
        stride_s: Step size for the sliding window. If None, equals window_size_s (non-overlapping).
        
    Returns:
        DataFrame where each row is a time window S_t.
    """
    if aligned_df.empty:
        return pd.DataFrame()
        
    if stride_s is None:
        stride_s = window_size_s
        
    # Determine global time bounds
    min_time = aligned_df["start_time"].min()
    max_time = aligned_df["end_time"].max()
    
    # Identify numeric feature columns to aggregate
    exclude = {"flow_key", "src_ip", "dst_ip", "protocol", "timestamp", "label", "is_attack", "start_time", "end_time"}
    feature_cols = [c for c in aligned_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(aligned_df[c])]
    
    windows = []
    
    current_time = min_time
    while current_time < max_time:
        window_end = current_time + window_size_s
        
        # Find flows active in this window (overlap > 0)
        mask = (aligned_df["start_time"] < window_end) & (aligned_df["end_time"] > current_time)
        active_flows = aligned_df[mask]
        
        if active_flows.empty:
            # Empty window - could emit all zeros, but typically we skip or zero-pad later
            current_time += stride_s
            continue
            
        window_record = {
            "window_start": current_time,
            "window_end": window_end,
            "active_flow_count": len(active_flows)
        }
        
        # Aggregate numeric features across all active flows in this window
        # We compute sum, mean, and max for each feature to capture the distribution
        for col in feature_cols:
            vals = active_flows[col].dropna()
            if len(vals) > 0:
                window_record[f"{col}_sum"] = vals.sum()
                window_record[f"{col}_mean"] = vals.mean()
                window_record[f"{col}_max"] = vals.max()
            else:
                window_record[f"{col}_sum"] = 0.0
                window_record[f"{col}_mean"] = 0.0
                window_record[f"{col}_max"] = 0.0
                
        # Generate temporal label (Task 1.6)
        if "is_attack" in active_flows.columns:
            # If any flow is an attack, the window is under attack
            attack_flows = active_flows["is_attack"] == 1.0
            window_record["is_attack"] = 1.0 if attack_flows.any() else 0.0
            window_record["attack_flow_ratio"] = attack_flows.mean()
            
            # Map MITRE stage based on the string label if present
            if "label" in active_flows.columns:
                attack_labels = active_flows.loc[attack_flows, "label"]
                if not attack_labels.empty:
                    # Concatenate unique attack labels and map to MITRE
                    unique_labels = attack_labels.unique()
                    
                    try:
                        from src.data.mitre_mapper import get_mitre_mapping, format_mitre_string
                        mitre_strs = [format_mitre_string(get_mitre_mapping(str(l))) for l in unique_labels]
                        window_record["attack_labels"] = " | ".join(mitre_strs)
                    except ImportError:
                        # Fallback if mapper not found
                        window_record["attack_labels"] = "|".join(str(l) for l in unique_labels)
                else:
                    window_record["attack_labels"] = "BENIGN"
                    
        windows.append(window_record)
        current_time += stride_s
        
    df_out = pd.DataFrame(windows)
    LOGGER.info("Aggregated %d flows into %d windows (size=%.1fs, stride=%.1fs)", 
                len(aligned_df), len(df_out), window_size_s, stride_s)
    return df_out


def _selftest():
    df = pd.DataFrame([
        {
            "start_time": 0.0, "end_time": 5.0, 
            "byte_count": 100, "is_attack": 0, "label": "BENIGN"
        },
        {
            "start_time": 3.0, "end_time": 12.0, 
            "byte_count": 500, "is_attack": 1, "label": "DoS Hulk"
        }
    ])
    
    # 10 second windows
    res = aggregate_windows(df, window_size_s=10.0)
    assert len(res) == 2
    
    w1 = res.iloc[0] # 0-10s
    assert w1["active_flow_count"] == 2
    assert w1["is_attack"] == 1.0
    assert w1["attack_labels"] == "Impact (T1498: Network Denial of Service)"
    assert w1["byte_count_sum"] == 600
    
    w2 = res.iloc[1] # 10-20s
    assert w2["active_flow_count"] == 1
    assert w2["byte_count_sum"] == 500
    
    print("temporal_aggregator selftest: OK")

if __name__ == "__main__":
    _selftest()
