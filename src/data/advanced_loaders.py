
import pandas as pd
import numpy as np

# Enterprise Unified Feature Space (10 Core Dimensions)
FEATURES = [
    "duration", "packet_count", "byte_count", 
    "flag_syn", "flag_ack", "flag_rst", 
    "auth_success", "auth_failure", "privilege_escalation",
    "is_network_event"
]

def load_ciciot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(0.0, index=np.arange(len(df)), columns=FEATURES)
    out["duration"] = df["flow_duration"]
    out["packet_count"] = df.get("Number", df["Header_Length"])
    out["byte_count"] = df.get("Tot sum", df.get("Tot size", 0))
    out["flag_syn"] = df.get("syn_count", 0)
    out["flag_ack"] = df.get("ack_count", 0)
    out["is_network_event"] = 1.0
    
    out["label"] = df["label"]
    out["is_attack"] = (df["label"] != "BenignTraffic").astype(float)
    return out

def load_darpa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(0.0, index=np.arange(len(df)), columns=FEATURES)
    out["duration"] = df["duration"]
    out["byte_count"] = df["src_bytes"] + df["dst_bytes"]
    out["packet_count"] = df["count"]
    out["flag_syn"] = (df["flag"] == "S0").astype(float)
    out["auth_failure"] = df["num_failed_logins"]
    out["privilege_escalation"] = df["num_compromised"]
    out["is_network_event"] = 1.0
    
    out["label"] = df["label"]
    out["is_attack"] = (df["label"] != "normal.").astype(float)
    return out

def load_lanl(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(0.0, index=np.arange(len(df)), columns=FEATURES)
    out["packet_count"] = 1.0 # 1 auth event
    
    out["auth_success"] = (df["success_failure"] == "Success").astype(float)
    out["auth_failure"] = (df["success_failure"] == "Fail").astype(float)
    out["privilege_escalation"] = (df["source_user"] == "U_Admin").astype(float)
    out["is_network_event"] = 0.0
    
    out["label"] = df["label"]
    out["is_attack"] = (df["label"] != "Normal").astype(float)
    return out

