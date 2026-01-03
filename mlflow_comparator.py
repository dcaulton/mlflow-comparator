import mlflow
import os
import tempfile
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO

# Config
MLFLOW_TRACKING_URI = "http://mlflow-service.mlflow.svc.cluster.local:5000"  # k8s internal
EXPERIMENT_NAME = "exp-2026-yolo-vit"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if not exp:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
exp_id = exp.experiment_id

# Pull finished runs
runs = client.search_runs(
    experiment_ids=[exp_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"]
)

if not runs:
    print("No runs found. Exiting.")
    exit(0)

# Build DataFrame
data = []
for run in runs:
    data.append({
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "detector": run.data.params.get("detector_type", "unknown"),
        "inference_time": run.data.metrics.get("inference_time", None),
        "num_detections": run.data.metrics.get("num_detections", None),
        "top_confidence": run.data.metrics.get("top_confidence", None),
    })

df = pd.DataFrame(data)

# Group and filter complete pairs
grouped = df.groupby("run_name")

multi_groups = []
for name, group in grouped:
    detectors = sorted(group["detector"].dropna().unique())
    if len(detectors) >= 2:  # At least 2 for comparison
        multi_groups.append((name, group, detectors))

if not multi_groups:
    print("No events with 2+ detectors found. Exiting.")
    exit(0)

# Build multi_df safely
records = []
for name, group, detectors in multi_groups:
    row = {"run_name": name, "detectors": ",".join(detectors)}
    for _, r in group.iterrows():
        det = r["detector"]
        row[f"inference_time_{det}"] = r["inference_time"]
        row[f"num_detections_{det}"] = r["num_detections"]
        row[f"top_confidence_{det}"] = r["top_confidence"]
    records.append(row)

multi_df = pd.DataFrame(records)

# Set baseline (default to first detector if not set)
baseline = os.getenv("BASELINE_DETECTOR", multi_df["detectors"].iloc[0].split(",")[0])

# Compute diffs relative to baseline (only if column exists)
metrics = ["inference_time", "num_detections", "top_confidence"]
stats = {"num_events": len(multi_df), "detectors": multi_df["detectors"].iloc[0]}

for metric in metrics:
    baseline_col = f"{metric}_{baseline}"
    if baseline_col not in multi_df.columns:
        continue  # Skip if baseline missing for this metric
    
    for det in [c.split(f"{metric}_")[1] for c in multi_df.columns if c.startswith(f"{metric}_") and c != baseline_col]:
        det_col = f"{metric}_{det}"
        diff_col = f"{metric}_diff_vs_{det}"
        multi_df[diff_col] = multi_df[det_col] - multi_df[baseline_col]
        
        # Stats
        stats[f"mean_{metric}_{det}"] = multi_df[det_col].mean()
        stats[f"mean_{metric}_diff_vs_{det}"] = multi_df[diff_col].mean()
        if metric in ["num_detections", "top_confidence"]:
            stats[f"{det}_better_{metric}_pct"] = (multi_df[diff_col] > 0).mean() * 100

# Charts (now using flat columns)
with tempfile.TemporaryDirectory() as tmpdir:
    # Boxplot per metric
    for metric in metrics:
        cols = [c for c in multi_df.columns if c.startswith(metric) and not c.endswith("diff")]
        if len(cols) < 2:
            continue
        melt = pd.melt(multi_df, value_vars=cols, var_name="Detector", value_name=metric)
        melt["Detector"] = melt["Detector"].str.replace(f"{metric}_", "")
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=melt, x="Detector", y=metric)
        ax.set_title(f"{metric.replace('_', ' ').capitalize()} (n={len(multi_df)} events)")
        fig.savefig(os.path.join(tmpdir, f"boxplot_{metric}.png"), bbox_inches='tight')
        plt.close(fig)

    # Pairwise scatter for num_detections (baseline vs each other)
    det_cols = [c for c in multi_df.columns if c.startswith("num_detections_") and not c.endswith(baseline)]
    for col in det_cols:
        other_det = col.split("num_detections_")[1]
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=multi_df, x=f"num_detections_{baseline}", y=col, alpha=0.7)
        max_val = multi_df[[f"num_detections_{baseline}", col]].max().max()
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_title(f"Detections: {baseline.upper()} vs {other_det.upper()}")
        fig.savefig(os.path.join(tmpdir, f"scatter_dets_{baseline}_vs_{other_det}.png"), bbox_inches='tight')
        plt.close(fig)

    # Log summary run
    summary_name = f"summary-{datetime.now().strftime('%Y-%m-%d-%H%M')}"
    with mlflow.start_run(experiment_id=exp_id, run_name=summary_name) as run:
        mlflow.log_param("num_events", len(multi_df))
        mlflow.log_param("baseline_detector", baseline)
        mlflow.log_param("all_detectors", stats["detectors"])

        for k, v in stats.items():
            if isinstance(v, (int, float)) and pd.notna(v):
                mlflow.log_metric(k, float(v))

        # Fixed: pass DataFrames directly
        mlflow.log_table(multi_df, "multi_event_summary.json")
        mlflow.log_table(multi_df.describe(), "aggregate_stats.json")

        for f in os.listdir(tmpdir):
            mlflow.log_artifact(os.path.join(tmpdir, f))

        print(f"Summary run '{summary_name}' created successfully.")
