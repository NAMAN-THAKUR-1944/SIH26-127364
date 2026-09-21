"""MITRE ATT&CK Mapping Utility.

Translates raw dataset anomaly labels (e.g., CTU-13, CIC-IDS2017, UNSW-NB15) 
into standardized MITRE ATT&CK Tactics and Techniques.
"""

from typing import Dict

# Comprehensive mapping of common dataset labels to MITRE ATT&CK stages
MITRE_MAPPING: Dict[str, Dict[str, str]] = {
    # CIC-IDS2017
    "DoS Hulk": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    "DoS GoldenEye": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    "DoS slowloris": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    "DoS Slowhttptest": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    "DDoS": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    "PortScan": {"tactic": "Discovery", "technique": "T1046", "name": "Network Service Discovery"},
    "Bot": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Infiltration": {"tactic": "Initial Access", "technique": "T1190", "name": "Exploit Public-Facing Application"},
    "Web Attack \xbf Brute Force": {"tactic": "Credential Access", "technique": "T1110", "name": "Brute Force"},
    "Web Attack \xbf XSS": {"tactic": "Initial Access", "technique": "T1190", "name": "Exploit Public-Facing Application"},
    "Web Attack \xbf Sql Injection": {"tactic": "Initial Access", "technique": "T1190", "name": "Exploit Public-Facing Application"},
    "FTP-Patator": {"tactic": "Credential Access", "technique": "T1110", "name": "Brute Force"},
    "SSH-Patator": {"tactic": "Credential Access", "technique": "T1110", "name": "Brute Force"},
    
    # CTU-13 (Botnets)
    "Botnet": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Rbot": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Virut": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Murlo": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Menti": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Neris": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Donbot": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Botnet Sogou": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    
    # CICIoT2023
    "Mirai": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    
    # DARPA
    "neptune": {"tactic": "Impact", "technique": "T1498", "name": "Network Denial of Service"},
    
    # LANL
    "Lateral Movement": {"tactic": "Lateral Movement", "technique": "T1550", "name": "Use Alternate Authentication Material"},
    
    # UNSW-NB15
    "Fuzzers": {"tactic": "Discovery", "technique": "T1046", "name": "Network Service Discovery"},
    "Analysis": {"tactic": "Discovery", "technique": "T1046", "name": "Network Service Discovery"},
    "Backdoors": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
    "Exploits": {"tactic": "Initial Access", "technique": "T1190", "name": "Exploit Public-Facing Application"},
    "Generic": {"tactic": "Execution", "technique": "T1059", "name": "Command and Scripting Interpreter"},
    "Reconnaissance": {"tactic": "Reconnaissance", "technique": "T1595", "name": "Active Scanning"},
    "Shellcode": {"tactic": "Execution", "technique": "T1059", "name": "Command and Scripting Interpreter"},
    "Worms": {"tactic": "Lateral Movement", "technique": "T1210", "name": "Exploitation of Remote Services"},
    
    "BENIGN": {"tactic": "None", "technique": "None", "name": "None"},
    "Normal": {"tactic": "None", "technique": "None", "name": "None"},
}

def get_mitre_mapping(raw_label: str) -> Dict[str, str]:
    """Look up the MITRE tactic and technique for a raw dataset label.
    
    Falls back to a generic 'Unknown' mapping if the label is not found.
    """
    # Try exact match
    if raw_label in MITRE_MAPPING:
        return MITRE_MAPPING[raw_label]
        
    # Try case-insensitive fallback
    for k, v in MITRE_MAPPING.items():
        if k.lower() in raw_label.lower():
            return v
        if raw_label.lower() in k.lower():
            return v
            
    # Default fallback
    return {"tactic": "Unknown", "technique": "Unknown", "name": "Unmapped Anomaly"}

def format_mitre_string(mapping: Dict[str, str]) -> str:
    """Format mapping dict into a readable string (e.g., 'Impact (T1498)')"""
    if mapping["tactic"] == "None":
        return "BENIGN"
    return f"{mapping['tactic']} ({mapping['technique']}: {mapping['name']})"

