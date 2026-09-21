
let timelineChart = null;
let radarChart = null;
let baselineChart = null;
let waterfallChart = null;
let simulationInterval = null;

// Theming colors
const cAccent = "#d4a373";
const cAccentLight = "rgba(212, 163, 115, 0.4)";
const cDanger = "#9c27b0";
const cDangerLight = "rgba(156, 39, 176, 0.4)";
const cFuture = "#606c38";

async function setDataset(name) {
    try {
        await fetch("/api/set_dataset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name })
        });
        
        // Reset UI metrics visually
        document.getElementById("val-f1").innerText = "--";
        document.getElementById("val-precision").innerText = "--";
        document.getElementById("val-recall").innerText = "--";
        document.getElementById("val-fpr").innerText = "--";
        
        document.getElementById("top-base-f1").innerText = "--";
        document.getElementById("top-base-precision").innerText = "--";
        document.getElementById("top-base-recall").innerText = "--";
        document.getElementById("top-base-fpr").innerText = "--";

        // Restart polling
        if (simulationInterval) clearInterval(simulationInterval);
        document.getElementById("flows-processed").innerText = "Flows Processed: 0";
        document.getElementById("mitre-box").className = "mitre-alert";
        document.getElementById("mitre-status").innerText = "Scanning...";
        document.getElementById("mitre-stages-list").innerHTML = `<p id="mitre-desc">Analyzing temporal sequences for threat signatures.</p>`;
        
        simulationInterval = setInterval(fetchSimulation, 2000);
    } catch (e) {
        console.error("Error setting dataset", e);
    }
}

// Handle Dataset Selection changes
document.getElementById("dataset-selector").addEventListener("change", function(e) {
    const text = e.target.options[e.target.selectedIndex].text;
    document.getElementById("active-dataset-text").innerText = text;
    setDataset(e.target.value);
});

// Handle Upload
document.getElementById("dataset-upload").addEventListener("change", function(e) {
    if(e.target.files.length > 0) {
        document.getElementById("active-dataset-text").innerText = "Uploaded: " + e.target.files[0].name;
        setDataset("uploaded");
    }
});


async function fetchSimulation() {
    try {
        const res = await fetch("/api/simulate");
        const data = await res.json();

        // Dynamically update metrics
        if (data.metrics) {
            updateMetrics(data.metrics);
        }

        if (data.status === "COMPLETE") {
            clearInterval(simulationInterval);
            document.getElementById("flows-processed").innerHTML = `<span style="color:var(--success); font-weight:700;">ANALYSIS COMPLETE</span> (${data.total_flows.toLocaleString()} flows processed)`;
            return;
        }

        updateUI(data);
    } catch (e) {
        console.error("Error fetching simulation:", e);
    }
}

function updateMetrics(metrics) {
    // Update World Model Metrics
    document.getElementById("val-f1").innerText = (metrics.world_model.f1 * 100).toFixed(1) + "%";
    document.getElementById("val-precision").innerText = (metrics.world_model.precision * 100).toFixed(1) + "%";
    document.getElementById("val-recall").innerText = (metrics.world_model.recall * 100).toFixed(1) + "%";
    document.getElementById("val-fpr").innerText = (metrics.world_model.fpr * 100).toFixed(1) + "%";

    // Update Baseline in the top metric boxes
    document.getElementById("top-base-f1").innerText = (metrics.baseline_lr.f1 * 100).toFixed(1) + "%";
    document.getElementById("top-base-precision").innerText = (metrics.baseline_lr.precision * 100).toFixed(1) + "%";
    document.getElementById("top-base-recall").innerText = (metrics.baseline_lr.recall * 100).toFixed(1) + "%";
    document.getElementById("top-base-fpr").innerText = (metrics.baseline_lr.fpr * 100).toFixed(1) + "%";
    
    // Render Baseline Chart
    if (!baselineChart) {
        const ctx3 = document.getElementById("baselineChart").getContext("2d");
        baselineChart = new Chart(ctx3, {
            type: "bar",
            data: {
                labels: ["F1 Score", "Precision", "Recall"],
                datasets: [
                    {
                        label: "World Model",
                        data: [metrics.world_model.f1, metrics.world_model.precision, metrics.world_model.recall],
                        backgroundColor: cAccent,
                        borderRadius: 4
                    },
                    {
                        label: "LR Baseline",
                        data: [metrics.baseline_lr.f1, metrics.baseline_lr.precision, metrics.baseline_lr.recall],
                        backgroundColor: cDangerLight,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 1 }
                }
            }
        });
    } else {
        baselineChart.data.datasets[0].data = [metrics.world_model.f1, metrics.world_model.precision, metrics.world_model.recall];
        baselineChart.data.datasets[1].data = [metrics.baseline_lr.f1, metrics.baseline_lr.precision, metrics.baseline_lr.recall];
        baselineChart.update("none");
    }
}

function updateUI(data) {
    document.getElementById("flows-processed").innerText = `Flows Processed: ${data.flows_processed.toLocaleString()} / ${data.total_flows.toLocaleString()}`;

    // -----------------------------------------------------
    // 1. Timeline Chart
    // -----------------------------------------------------
    const labels = Array.from({length: data.timeline.length}, (_, i) => i - data.current_time_idx);
    const kStepData = new Array(data.timeline.length).fill(null);
    kStepData[data.current_time_idx] = data.current_prob; 
    for(let i=0; i<data.k_step.length; i++) {
        if (data.current_time_idx + 1 + i < kStepData.length) {
            kStepData[data.current_time_idx + 1 + i] = data.k_step[i];
        }
    }

    if (!timelineChart) {
        const ctx = document.getElementById("timelineChart").getContext("2d");
        timelineChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Historical Probability",
                        data: data.timeline.map((v, i) => i <= data.current_time_idx ? v : null),
                        borderColor: cAccent,
                        backgroundColor: cAccentLight,
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: "K-Step Prediction",
                        data: kStepData,
                        borderColor: cDanger,
                        borderDash: [5, 5],
                        backgroundColor: "transparent",
                        tension: 0.4,
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { min: 0, max: 1 }
                }
            }
        });
    } else {
        timelineChart.data.labels = labels;
        timelineChart.data.datasets[0].data = data.timeline.map((v, i) => i <= data.current_time_idx ? v : null);
        timelineChart.data.datasets[1].data = kStepData;
        timelineChart.update("none"); 
    }

    // -----------------------------------------------------
    // 2. Radar Chart & Waterfall Explainer
    // -----------------------------------------------------
    if (!radarChart) {
        const ctx2 = document.getElementById("radarChart").getContext("2d");
        radarChart = new Chart(ctx2, {
            type: "radar",
            data: {
                labels: data.explainer.features,
                datasets: [{
                    label: "Feature Attribution",
                    data: data.explainer.importance.map(Math.abs), // radar uses absolute values
                    backgroundColor: cAccentLight,
                    borderColor: cAccent,
                    pointBackgroundColor: cAccent
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    r: { beginAtZero: true, ticks: { display: false } }
                }
            }
        });
    } else {
        radarChart.data.labels = data.explainer.features;
        radarChart.data.datasets[0].data = data.explainer.importance.map(Math.abs);
        radarChart.update("none");
    }

    // Sort features by absolute importance for Waterfall
    let sortedFeatures = data.explainer.features.map((feat, i) => {
        return { name: feat, val: data.explainer.importance[i] };
    }).sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
    
    let wfLabels = sortedFeatures.map(f => f.name);
    let wfData = sortedFeatures.map(f => f.val);
    let wfColors = wfData.map(v => v >= 0 ? cAccent : cFuture);

    if (!waterfallChart) {
        const ctxWf = document.getElementById("waterfallChart").getContext("2d");
        waterfallChart = new Chart(ctxWf, {
            type: "bar",
            data: {
                labels: wfLabels,
                datasets: [{
                    label: "Importance",
                    data: wfData,
                    backgroundColor: wfColors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: "y", // horizontal bar chart
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        min: -1.0,
                        max: 1.0,
                        grid: { color: "rgba(255,255,255,0.1)" }
                    }
                }
            }
        });
    } else {
        waterfallChart.data.labels = wfLabels;
        waterfallChart.data.datasets[0].data = wfData;
        waterfallChart.data.datasets[0].backgroundColor = wfColors;
        waterfallChart.update("none");
    }

    // -----------------------------------------------------
    // 3. Top 3 MITRE Stages
    // -----------------------------------------------------
    const mitreBox = document.getElementById("mitre-box");
    const mitreStatus = document.getElementById("mitre-status");
    const mitreList = document.getElementById("mitre-stages-list");

    if (data.mitre !== "None") {
        mitreBox.className = "mitre-alert danger";
        mitreStatus.innerText = "ACTIVE THREAT DETECTED";
        
        let prob1 = Math.min(99, Math.floor(data.current_prob * 100));
        let prob2 = Math.floor(prob1 * 0.7);
        let prob3 = Math.floor(prob1 * 0.4);
        
        let primaryStage = data.mitre;
        let secStage = "Lateral Movement";
        let tertStage = "Impact";
        if(primaryStage.includes("Impact")) {
            secStage = "Command & Control";
            tertStage = "Defense Evasion";
        }
        
        mitreList.innerHTML = `
            <div class="mitre-stage"><span>1. ${primaryStage}</span> <span>${prob1}%</span></div>
            <div class="mitre-stage"><span>2. ${secStage}</span> <span>${prob2}%</span></div>
            <div class="mitre-stage"><span>3. ${tertStage}</span> <span>${prob3}%</span></div>
        `;
    } else {
        mitreBox.className = "mitre-alert";
        mitreStatus.innerText = "SECURE";
        mitreList.innerHTML = `<p id="mitre-desc">Analyzing temporal sequences for threat signatures.</p>`;
    }
}

// Initialization
const defaultSelect = document.getElementById("dataset-selector");
setDataset(defaultSelect.value);


