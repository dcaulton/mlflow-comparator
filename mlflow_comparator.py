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

# Collect multi-detector groups (no "complete_pairs" filter—handle all)
multi_groups = []
for name, group in grouped:
    detectors = sorted(group["detector"].unique())
    if len(detectors) >= 2:  # Min for comparison; adjust as needed
        multi_groups.append((name, group, detectors))

if not multi_groups:
    print("No multi-detector groups found. Exiting.")
    exit(0)

# Build expanded pairs_df → now "multi_df" with pivoted metrics
# Pivot for easy diffs (columns like inference_time_yolo, _vit, _newyolo)
pivot_metrics = ["inference_time", "num_detections", "top_confidence"]
multi_df = pd.DataFrame()  # Aggregate across groups
for name, group, detectors in multi_groups:
    pivot = group.pivot(index="run_name", columns="detector", values=pivot_metrics)
    pivot["run_name"] = name
    multi_df = pd.concat([multi_df, pivot])

# Compute diffs relative to a baseline (configurable)
baseline = os.getenv("BASELINE_DETECTOR", "yolo")  # Env var for flexibility
for metric in pivot_metrics:
    for det in [d for d in multi_df.columns.levels[1] if d != baseline]:
        multi_df[(metric, f"diff_{det}")] = multi_df[(metric, det)] - multi_df[(metric, baseline)]

# Aggregates (means/std across all, plus pairwise)
stats = {}
for metric in pivot_metrics:
    stats[f"mean_{metric}"] = multi_df[metric].mean(axis=1).mean()  # Overall mean
    stats[f"std_{metric}"] = multi_df[metric].std(axis=1).mean()   # Avg variation
    for det in multi_df[metric].columns:
        stats[f"mean_{metric}_{det}"] = multi_df[(metric, det)].mean()

# Add pairwise superiority % (e.g., % events where vit > yolo)
for metric in ["num_detections", "top_confidence"]:
    for det in [d for d in multi_df[metric].columns if d != baseline]:
        stats[f"{det}_higher_{metric}_pct"] = (multi_df[(metric, det)] > multi_df[(metric, baseline)]).mean() * 100

# Charts: Adapt to multi-detector
with tempfile.TemporaryDirectory() as tmpdir:
    # Boxplot for each metric (side-by-side for all detectors)
    for metric in pivot_metrics:
        melt_df = pd.melt(multi_df[metric].reset_index(), id_vars="run_name", var_name="Detector", value_name=metric.capitalize())
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=melt_df, x="Detector", y=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Distribution Across Detectors (n={len(multi_df)} events)")
        box_path = os.path.join(tmpdir, f"boxplot_{metric}.png")
        fig.savefig(box_path, bbox_inches='tight')
        plt.close(fig)

    # Scatter: Pairwise (loop over pairs, but limit to key ones like baseline vs each)
    for det in [d for d in multi_df["num_detections"].columns if d != baseline]:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=multi_df[("num_detections", baseline)], y=multi_df[("num_detections", det)])
        ax.plot([0, multi_df["num_detections"].max().max()], [0, multi_df["num_detections"].max().max()], 'r--')
        ax.set_title(f"Num Detections: {baseline.upper()} vs {det.upper()}")
        scatter_path = os.path.join(tmpdir, f"scatter_dets_{baseline}_vs_{det}.png")
        fig.savefig(scatter_path, bbox_inches='tight')
        plt.close(fig)

    # Similar for hist diffs (per pair)
    # ... (add as needed)
# Hist conf diff
#fig2, ax2 = plt.subplots(figsize=(8,6))
#sns.histplot(pairs_df["conf_diff"], kde=True)
#ax2.axvline(stats["mean_conf_diff"], color='r', linestyle='--')
#ax2.set_title("Top Confidence Diff (ViT - YOLO)")
#hist_buf = fig_to_buf(fig2)
#plt.close(fig2)
#
## Boxplot conf
#conf_df = pd.melt(pairs_df[["yolo_conf", "vit_conf"]], var_name="Detector", value_name="Top Confidence")
#conf_df["Detector"] = conf_df["Detector"].str.replace("_conf", "").str.upper()
#fig3, ax3 = plt.subplots(figsize=(8,6))
#sns.boxplot(data=conf_df, x="Detector", y="Top Confidence")
#ax3.set_title("Top Confidence Distribution")
#box_buf = fig_to_buf(fig3)
#plt.close(fig3)

    # Log to summary run
    summary_name = f"summary-{datetime.now().strftime('%Y-%m-%d-%H%M')}"
    with mlflow.start_run(experiment_id=exp_id, run_name=summary_name) as run:
        mlflow.log_param("num_events", len(multi_df))
        mlflow.log_param("detectors_included", ",".join(set(df["detector"])))
        mlflow.log_param("baseline_detector", baseline)

        for k, v in stats.items():
            if pd.notna(v):
                mlflow.log_metric(k, float(v))

        mlflow.log_table(multi_df.reset_index(), "multi_summary.json")
        mlflow.log_table(multi_df.describe(), "aggregate_stats.json")

        # Log all generated charts
        for chart_file in os.listdir(tmpdir):
            mlflow.log_artifact(os.path.join(tmpdir, chart_file))
        
        print(f"Logged summary run '{summary_name}' with {len(pairs_df)} pairs and [{len(os.listdir(tmpdir))}] charts.")

print("Comparator run complete.")
