"""Align PCAP flow features with CSV flow features chronologically.

PCAP data is the source of truth for time boundaries since it's derived directly
from packets. The CSV data (e.g. CIC-IDS) may have arbitrary aggregation windows.
This module aligns them based on the 5-tuple (flow_key) and calculates
overlap-weighted aggregates of the CSV features that fall within the PCAP's
[start_time, end_time] bounds.
"""

from __future__ import annotations

import logging
from typing import Optional, List
import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.data.flow_parser import OUTPUT_COLUMNS as CSV_COLUMNS

LOGGER = logging.getLogger(__name__)

# Additive features that should be scaled by overlap time fraction
_ADDITIVE_FEATURES = [
    "packet_count", "fwd_packets", "bwd_packets",
    "byte_count", "fwd_bytes", "bwd_bytes",
    "flag_fin", "flag_syn", "flag_rst", "flag_psh", 
    "flag_ack", "flag_urg", "flag_cwr", "flag_ece",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags"
]

# Features that should be aggregated via weighted mean (weighted by overlap time)
_MEAN_FEATURES = [
    c for c in CSV_COLUMNS if "_mean" in c or "_std" in c or "ratio" in c or "per_second" in c
]

# Features that should be aggregated via min
_MIN_FEATURES = [c for c in CSV_COLUMNS if "_min" in c]

# Features that should be aggregated via max
_MAX_FEATURES = [c for c in CSV_COLUMNS if "_max" in c]


def _build_pcap_flow_key(df: pd.DataFrame) -> pd.Series:
    """Build the canonical flow_key string for a pcap dataframe."""
    return df["src_ip"].astype(str) + "|" + \
           df["src_port"].astype(int).astype(str) + "|" + \
           df["dst_ip"].astype(str) + "|" + \
           df["dst_port"].astype(int).astype(str) + "|" + \
           df["protocol"].astype(str)


def align_telemetry(
    pcap_df: pd.DataFrame, 
    csv_df: pd.DataFrame, 
    tolerance_s: float = 1.0
) -> pd.DataFrame:
    """Align PCAP features with Flow CSV features.
    
    Args:
        pcap_df: DataFrame from pcap_parser.
        csv_df: DataFrame from flow_parser.
        tolerance_s: Clock skew tolerance in seconds.
        
    Returns:
        DataFrame of joined features.
    """
    if pcap_df.empty or csv_df.empty:
        LOGGER.warning("Empty dataframe provided to align_telemetry")
        return pd.DataFrame()
        
    # Ensure flow_key exists in both
    if "flow_key" not in pcap_df.columns:
        pcap_df = pcap_df.copy()
        pcap_df["flow_key"] = _build_pcap_flow_key(pcap_df)
        
    # Calculate end_time for CSV if not present
    if "end_time" not in csv_df.columns:
        csv_df = csv_df.copy()
        csv_df["end_time"] = csv_df["start_time"] + csv_df["duration"].fillna(0)

    # We will iterate through groups. For larger datasets, a vectorized interval join
    # using np.searchsorted or pandas.merge_asof is better, but this is a robust
    # fallback for prototype size.
    
    aligned_records = []
    
    csv_grouped = csv_df.groupby("flow_key")
    
    for idx, pcap_row in pcap_df.iterrows():
        key = pcap_row["flow_key"]
        if key not in csv_grouped.groups:
            # No matching flows in CSV
            continue
            
        csv_subset = csv_grouped.get_group(key)
        
        # Calculate overlap
        p_start = pcap_row["start_time"] - tolerance_s
        p_end = pcap_row["end_time"] + tolerance_s
        p_dur = max(p_end - p_start, 1e-6)
        
        overlaps = []
        for _, c_row in csv_subset.iterrows():
            c_start = c_row["start_time"]
            c_end = c_row["end_time"]
            c_dur = max(c_end - c_start, 1e-6)
            
            o_start = max(p_start, c_start)
            o_end = min(p_end, c_end)
            
            if o_start <= o_end:
                overlap_dur = o_end - o_start
                overlaps.append((c_row, overlap_dur, c_dur))
                
        if not overlaps:
            continue
            
        # Aggregate features
        record = pcap_row.to_dict()
        record["flow_match_count"] = len(overlaps)
        total_overlap = sum(o[1] for o in overlaps)
        record["flow_coverage"] = total_overlap / p_dur
        
        # Weighted scaling
        for feat in _ADDITIVE_FEATURES:
            if feat in csv_df.columns:
                val = sum((o[0][feat] * (o[1] / o[2])) for o in overlaps if not pd.isna(o[0][feat]))
                record[feat] = val
                
        for feat in _MEAN_FEATURES:
            if feat in csv_df.columns:
                valid_o = [o for o in overlaps if not pd.isna(o[0][feat])]
                if valid_o:
                    val = sum((o[0][feat] * o[1]) for o in valid_o) / sum(o[1] for o in valid_o)
                    record[feat] = val
                else:
                    record[feat] = np.nan
                    
        for feat in _MIN_FEATURES:
            if feat in csv_df.columns:
                valid_vals = [o[0][feat] for o in overlaps if not pd.isna(o[0][feat])]
                record[feat] = min(valid_vals) if valid_vals else np.nan
                
        for feat in _MAX_FEATURES:
            if feat in csv_df.columns:
                valid_vals = [o[0][feat] for o in overlaps if not pd.isna(o[0][feat])]
                record[feat] = max(valid_vals) if valid_vals else np.nan
                
        # Label is logical OR (if any overlap is attack, flow is attack)
        if "is_attack" in csv_df.columns:
            valid_vals = [o[0]["is_attack"] for o in overlaps if not pd.isna(o[0]["is_attack"])]
            record["is_attack"] = max(valid_vals) if valid_vals else np.nan
            
        aligned_records.append(record)
        
    if not aligned_records:
        return pd.DataFrame()
        
    return pd.DataFrame(aligned_records)


def _selftest() -> None:
    """Verify vectorised canonicalisation and overlap logic."""
    from src.data.pcap_parser import _canonical_key as pcap_key
    from src.data.flow_parser import _canonical_key as flow_key
    
    # 1. Test canonical keys match
    assert pcap_key("10.0.0.1", 1234, "10.0.0.2", 80, "TCP")[0] == flow_key("10.0.0.1", 1234, "10.0.0.2", 80, "TCP")
    assert pcap_key("10.0.0.1", 1234, "10.0.0.2", 80, "TCP")[0] == flow_key("10.0.0.2", 80, "10.0.0.1", 1234, "TCP")
    
    # 2. Test alignment logic
    pcap = pd.DataFrame([{
        "src_ip": "10.0.0.1", "src_port": 1234, 
        "dst_ip": "10.0.0.2", "dst_port": 80, 
        "protocol": "TCP",
        "start_time": 10.0, "end_time": 20.0, "duration": 10.0
    }])
    
    csv = pd.DataFrame([{
        "flow_key": "10.0.0.1|1234|10.0.0.2|80|TCP",
        "start_time": 15.0, "duration": 10.0,  # ends at 25.0, 50% overlap with pcap (15 to 20)
        "byte_count": 1000, "is_attack": 1.0,
        "flow_iat_mean": 5.0
    }])
    
    df = align_telemetry(pcap, csv, tolerance_s=0.0)
    assert len(df) == 1
    assert df.iloc[0]["flow_match_count"] == 1
    assert df.iloc[0]["flow_coverage"] == 0.5  # 5s overlap / 10s pcap duration
    assert df.iloc[0]["byte_count"] == 500  # 1000 * 5/10 (scaled by its own duration overlap)
    assert df.iloc[0]["is_attack"] == 1.0
    assert df.iloc[0]["flow_iat_mean"] == 5.0
    
    print("telemetry_aligner selftest: OK")


if __name__ == "__main__":
    _selftest()
