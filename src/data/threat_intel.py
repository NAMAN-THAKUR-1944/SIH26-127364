
from typing import Dict
from src.data.mitre_mapper import get_mitre_mapping, format_mitre_string

# CAPEC (Common Attack Pattern Enumeration and Classification) mappings
CAPEC_MAPPING = {
    "Network Denial of Service": "CAPEC-125 (Flooding)",
    "Network Service Discovery": "CAPEC-277 (Data Footprinting)",
    "Application Layer Protocol": "CAPEC-268 (Audit Log Manipulation)",
    "Exploit Public-Facing Application": "CAPEC-66 (SQL Injection)",
    "Brute Force": "CAPEC-112 (Brute Force)",
    "Lateral Movement": "CAPEC-561 (Pass the Hash)"
}

# Vulnerability Database (CVE/NVD) mock mappings
CVE_MAPPING = {
    "CAPEC-66 (SQL Injection)": ["CVE-2021-44228 (Log4j)", "CVE-2023-23397"],
    "CAPEC-125 (Flooding)": ["CVE-2023-44487 (HTTP/2 Rapid Reset)"],
    "CAPEC-561 (Pass the Hash)": ["CVE-2020-1472 (ZeroLogon)"]
}

def enrich_threat_intel(raw_label: str) -> Dict[str, str]:
    """Enriches a raw dataset label with MITRE, CAPEC, and CVE intel."""
    
    # 1. Base MITRE Mapping
    mitre = get_mitre_mapping(raw_label)
    
    if mitre["name"] == "None":
        return {"threat": "BENIGN", "capec": "None", "cve": []}
        
    mitre_str = format_mitre_string(mitre)
    
    # 2. CAPEC Mapping
    capec = CAPEC_MAPPING.get(mitre["name"], "Unknown CAPEC")
    
    # 3. CVE Integration
    cves = CVE_MAPPING.get(capec, [])
    
    return {
        "threat": mitre_str,
        "capec": capec,
        "cve": cves
    }

