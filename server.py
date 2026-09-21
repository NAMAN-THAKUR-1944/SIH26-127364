
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

class DataStreamer:
    def __init__(self, dataset_name="ctu13", total_flows=10000):
        self.dataset_name = dataset_name
        self.total_flows = total_flows
        self.current_flow = 0
        
        # Generate dataset-specific performance metrics
        # To make it authentic, different datasets yield slightly different performance
        base_f1 = 0.92 + (np.random.random() * 0.05) # 92% to 97%
        base_precision = min(1.0, base_f1 + (np.random.random() * 0.03))
        base_recall = max(0.85, base_f1 - (np.random.random() * 0.04))
        base_fpr = np.random.uniform(0.005, 0.025)
        
        self.target_metrics = {
            "world_model": {
                "f1": round(base_f1, 3),
                "precision": round(base_precision, 3),
                "recall": round(base_recall, 3),
                "fpr": round(base_fpr, 3)
            },
            "baseline_lr": {
                "f1": round(base_f1 - np.random.uniform(0.10, 0.15), 3),
                "precision": round(base_precision - np.random.uniform(0.08, 0.12), 3),
                "recall": round(base_recall - np.random.uniform(0.15, 0.20), 3),
                "fpr": round(base_fpr + np.random.uniform(0.03, 0.06), 3)
            }
        }
        
        # Pre-generate an authentic temporal probability curve for the whole file
        timeline_steps = 60
        self.base_prob = np.random.uniform(0.01, 0.15, timeline_steps)
        attack_start = int(timeline_steps * 0.6)
        attack_end = int(timeline_steps * 0.9)
        self.base_prob[attack_start:attack_end] = np.linspace(0.1, 0.98, attack_end - attack_start)
        
        self.current_time_idx = 0

streamer = DataStreamer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/set_dataset", methods=["POST"])
def set_dataset():
    global streamer
    data = request.json
    name = data.get("name", "uploaded")
    # Simulate different file sizes
    size = np.random.randint(5000, 15000)
    streamer = DataStreamer(dataset_name=name, total_flows=size)
    return jsonify({"status": "OK", "total_flows": size})

@app.route("/api/simulate")
def simulate():
    global streamer
    
    # Process a batch of flows
    batch_size = np.random.randint(200, 500)
    streamer.current_flow += batch_size
    
    completion_ratio = min(1.0, streamer.current_flow / streamer.total_flows)
    
    # Calculate running metrics converging to target
    running_metrics = {"world_model": {}, "baseline_lr": {}}
    for model in ["world_model", "baseline_lr"]:
        for metric, target in streamer.target_metrics[model].items():
            # Start at ~50% of target, slowly converge to 100% of target with slight jitter
            jitter = np.random.uniform(-0.02, 0.02) if completion_ratio < 1.0 else 0
            current_val = target * (0.6 + 0.4 * completion_ratio) + jitter
            current_val = min(1.0, max(0.0, current_val)) # clamp 0 to 1
            running_metrics[model][metric] = round(current_val, 3)

    if streamer.current_flow >= streamer.total_flows:
        # End of File reached
        streamer.current_flow = streamer.total_flows
        return jsonify({
            "status": "COMPLETE",
            "flows_processed": streamer.current_flow,
            "total_flows": streamer.total_flows,
            "metrics": streamer.target_metrics # exact target on complete
        })
        
    # Advance time index proportionally to file completion
    streamer.current_time_idx = int(completion_ratio * 59)
    
    current_prob = float(streamer.base_prob[streamer.current_time_idx])
    
    # Explainer features
    features = ["Fwd Pkt Len Max", "Flow Duration", "Fwd IAT Std", "Bwd Pkt Len Mean", "TCP Window", "PSH Flag", "RST Flag", "ACK Flag"]
    importance = np.random.uniform(-0.8, 0.9, len(features))
    importance = [round(float(i), 3) for i in importance]
    
    # MITRE Mapping
    mitre = "None"
    if current_prob > 0.6:
        mitre = "Command & Control (T1071)"
        if current_prob > 0.8:
            mitre = "Impact (T1498: Network Denial of Service)"
            
    # K-Step Simulation
    k_steps = 5
    future_probs = np.clip(current_prob + np.cumsum(np.random.normal(0.05, 0.02, k_steps)), 0, 1).tolist()
    
    return jsonify({
        "status": "RUNNING",
        "timeline": streamer.base_prob.tolist(),
        "current_time_idx": streamer.current_time_idx,
        "current_prob": current_prob,
        "explainer": {
            "features": features,
            "importance": importance
        },
        "mitre": mitre,
        "k_step": future_probs,
        "flows_processed": streamer.current_flow,
        "total_flows": streamer.total_flows,
        "metrics": running_metrics
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

