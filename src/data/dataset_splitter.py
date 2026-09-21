"""Dataset Splitting configuration.

Provides functions to split datasets into train, validation, and hold-out
evaluation sets. For the world model evaluation, we hold out specific
attack patterns entirely from the training set to evaluate unseen-attack
generalization capabilities (Task 2.3).
"""

import logging
from typing import Tuple, List
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Specific attack labels (sub-types) that will NEVER be seen during training.
# They are strictly held out to evaluate true generalization to zero-day attacks.
HOLD_OUT_ATTACKS = {
    "Botnet",            # From CTU-13 / CIC
    "Web Attack",        # Cross-site scripting, SQLi etc.
    "Infiltration"
}

def split_dataset(
    df: pd.DataFrame, 
    train_frac: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation, and hold-out zero-day evaluation sets.
    
    Args:
        df: The aggregated temporal state windows DataFrame.
        train_frac: Fraction of the remaining data (after hold-out) for training.
        random_state: Seed for reproducibility.
        
    Returns:
        (train_df, val_df, holdout_df)
    """
    if df.empty:
        return df, df, df
        
    df = df.copy()
    
    # 1. Isolate hold-out attacks
    holdout_mask = pd.Series(False, index=df.index)
    if "attack_labels" in df.columns:
        for attack in HOLD_OUT_ATTACKS:
            # Case insensitive substring match
            mask = df["attack_labels"].astype(str).str.contains(attack, case=False, na=False)
            holdout_mask = holdout_mask | mask
            
    holdout_df = df[holdout_mask]
    remaining_df = df[~holdout_mask]
    
    # 2. Split the remainder into train and val chronologically or randomly
    # Since it's temporal, a chronological split might be preferred, but 
    # for standard baselines we'll shuffle. Let's do random split for the prototype.
    train_df = remaining_df.sample(frac=train_frac, random_state=random_state)
    val_df = remaining_df.drop(train_df.index)
    
    LOGGER.info(
        "Dataset Split -> Train: %d, Val: %d, Zero-Day Hold-out: %d",
        len(train_df), len(val_df), len(holdout_df)
    )
    
    return train_df, val_df, holdout_df
    
def _selftest() -> None:
    # Synthetic data
    df = pd.DataFrame([
        {"window_start": 0, "is_attack": 0, "attack_labels": "BENIGN"},
        {"window_start": 10, "is_attack": 0, "attack_labels": "BENIGN"},
        {"window_start": 20, "is_attack": 1, "attack_labels": "DoS Hulk"}, # Should go to train/val
        {"window_start": 30, "is_attack": 1, "attack_labels": "Botnet A"}, # Should go to holdout
        {"window_start": 40, "is_attack": 1, "attack_labels": "Infiltration Attempt"} # Should go to holdout
    ])
    
    train, val, holdout = split_dataset(df, train_frac=0.6)
    
    assert len(holdout) == 2
    assert "Botnet A" in holdout["attack_labels"].values
    assert "Infiltration Attempt" in holdout["attack_labels"].values
    
    assert len(train) + len(val) == 3
    assert len(train) == 2  # round(3 * 0.6)
    assert len(val) == 1
    
    print("dataset_splitter selftest: OK")
    
if __name__ == "__main__":
    _selftest()
