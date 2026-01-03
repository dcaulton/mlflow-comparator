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
complete_pairs = [ (name, group) for name, group in grouped if set(group["detector"]) == {"yolo", "vit"} and len(group) == 2 ]

if not complete_pairs:
    print("No complete pairs found. Exiting.")
    exit(0)

# Build pairs_df
pairs_data = []
for name, group in complete_pairs:
    yolo = group[group["detector"] == "yolo"].iloc[0]
    vit = group[group["detector"] == "vit"].iloc[0]
    pairs_data.append({
        "run_name": name,
        "yolo_dets": yolo["num_detections"],
        "vit_dets": vit["num_detections"],
        "yolo_conf": yolo["top_confidence"],
        "vit_conf": vit["top_confidence"],
        "yolo_time": yolo["inference_time"],
        "vit_time": vit["inference_time"],
    })
pairs_df = pd.DataFrame(pairs_data)

# Compute aggregates
pairs_df["conf_diff"] = pairs_df["vit_conf"] - pairs_df["yolo_conf"]
pairs_df["dets_diff"] = pairs_df["vit_dets"] - pairs_df["yolo_dets"]
pairs_df["time_diff"] = pairs_df["vit_time"] - pairs_df["yolo_time"]

stats = {
    "mean_conf_diff": pairs_df["conf_diff"].mean(),
    "std_conf_diff": pairs_df["conf_diff"].std(),
    "mean_dets_diff": pairs_df["dets_diff"].mean(),
    "std_dets_diff": pairs_df["dets_diff"].std(),
    "mean_time_diff": pairs_df["time_diff"].mean(),
    "std_time_diff": pairs_df["time_diff"].std(),
    "vit_more_dets_pct": (pairs_df["dets_diff"] > 0).mean() * 100,
    "vit_higher_conf_pct": (pairs_df["conf_diff"] > 0).mean() * 100,
}

# Generate charts as buffers
def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# Scatter detections
fig1, ax1 = plt.subplots(figsize=(8,6))
sns.scatterplot(data=pairs_df, x="yolo_dets", y="vit_dets")
ax1.plot([0, pairs_df[["yolo_dets","vit_dets"]].max().max()], [0, pairs_df[["yolo_dets","vit_dets"]].max().max()], 'r--')
ax1.set_title(f"Num Detections: YOLO vs ViT (n={len(pairs_df)})")
scatter_buf = fig_to_buf(fig1)
plt.close(fig1)

# Hist conf diff
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.histplot(pairs_df["conf_diff"], kde=True)
ax2.axvline(stats["mean_conf_diff"], color='r', linestyle='--')
ax2.set_title("Top Confidence Diff (ViT - YOLO)")
hist_buf = fig_to_buf(fig2)
plt.close(fig2)

# Boxplot conf
conf_df = pd.melt(pairs_df[["yolo_conf", "vit_conf"]], var_name="Detector", value_name="Top Confidence")
conf_df["Detector"] = conf_df["Detector"].str.replace("_conf", "").str.upper()
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.boxplot(data=conf_df, x="Detector", y="Top Confidence")
ax3.set_title("Top Confidence Distribution")
box_buf = fig_to_buf(fig3)
plt.close(fig3)

# Use a temporary directory to save charts as real files
with tempfile.TemporaryDirectory() as tmpdir:
    scatter_path = os.path.join(tmpdir, "scatter_detections.png")
    hist_path = os.path.join(tmpdir, "hist_conf_diff.png")
    box_path = os.path.join(tmpdir, "boxplot_conf.png")

    fig1.savefig(scatter_path, bbox_inches='tight', dpi=150)
    plt.close(fig1)

    fig2.savefig(hist_path, bbox_inches='tight', dpi=150)
    plt.close(fig2)

    fig3.savefig(box_path, bbox_inches='tight', dpi=150)
    plt.close(fig3)

    # Now start the MLflow run and log everything
    summary_name = f"summary-{datetime.now().strftime('%Y-%m-%d-%H%M')}"
    with mlflow.start_run(experiment_id=exp_id, run_name=summary_name) as run:
        mlflow.log_param("num_pairs", len(pairs_df))
        mlflow.log_param("summary_date", datetime.now().isoformat())

        for k, v in stats.items():
            if v is not None and not pd.isna(v):  # Extra guard against NaN
                mlflow.log_metric(k, float(v))

        # Fixed: Pass DataFrames directly
        mlflow.log_table(pairs_df, "pairs_summary.json")
        mlflow.log_table(pairs_df.describe(), "aggregate_stats.json")

        # Artifacts (unchanged)
        mlflow.log_artifact(scatter_path)
        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(box_path)

        print(f"Logged summary run '{summary_name}' with {len(pairs_df)} pairs and 3 charts.")

print("Comparator run complete.")
