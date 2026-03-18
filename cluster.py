"""
Part 2 – K-Means Clustering
Groups high schools by sports-infrastructure density (n_facilities, total_acres).

Input : school_features.csv  (from preprocess.py)
Output: school_clusters.csv
        cluster_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

INPUT_FILE  = "school_features.csv"
OUTPUT_CSV  = "school_clusters.csv"
OUTPUT_PLOT = "cluster_plot.png"

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} schools")

# Features for clustering: facility count and green space near each school
FEATURE_COLS = ["n_facilities", "total_acres"]
X = df[FEATURE_COLS].fillna(0).to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── Choose best k by silhouette score ─────────────────────────────────────────

print("\nSilhouette scores by k:")
best_k, best_score, best_model = 2, -1, None
max_k = min(6, len(df) - 1)

for k in range(2, max_k + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  k={k}  silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score, best_model = k, score, km

print(f"\n→ Best k = {best_k}  (silhouette = {best_score:.3f})")


# ── Assign clusters and label them ────────────────────────────────────────────

df["cluster"] = best_model.labels_

# Rank clusters by mean facility count (most → "Highly Developed")
cluster_means = (
    df.groupby("cluster")["n_facilities"]
    .mean()
    .sort_values(ascending=False)
)

rank_labels = ["High Access Zone", "Mid Access Zone", "Low Access Zone",
               "Zone 4", "Zone 5", "Zone 6"]

rank_map   = {cid: i for i, cid in enumerate(cluster_means.index)}
label_map  = {cid: rank_labels[i] for i, cid in enumerate(cluster_means.index)}
df["cluster_label"] = df["cluster"].map(label_map)

print("\nCluster summary (mean values):")
summary = df.groupby("cluster_label")[
    FEATURE_COLS + ["total_titles", "athletes", "participation_pct"]
].mean().round(2)
print(summary.to_string())


# ── Scatter plot: n_facilities vs total_acres, coloured by cluster ────────────

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]
labels_ordered = list(dict.fromkeys(
    label_map[cid] for cid in cluster_means.index
))

fig, ax = plt.subplots(figsize=(10, 7))

for i, label in enumerate(labels_ordered):
    mask = df["cluster_label"] == label
    ax.scatter(
        df.loc[mask, "n_facilities"],
        df.loc[mask, "total_acres"],
        color=COLORS[i % len(COLORS)],
        s=120, edgecolors="white", linewidths=0.8,
        zorder=3, label=label
    )

# Annotate school names
for _, row in df.iterrows():
    ax.annotate(
        row["school_name"].split("(")[0].strip(),   # shorten long names
        (row["n_facilities"], row["total_acres"]),
        fontsize=7.5, alpha=0.85,
        xytext=(5, 4), textcoords="offset points"
    )

ax.set_xlabel("Nearby Recreation Facilities (count within 1 mile)", fontsize=12)
ax.set_ylabel("Nearby Green Space (total acres within 1 mile)", fontsize=12)
ax.set_title(
    "K-Means Clustering of SF High Schools\nby Sports Infrastructure Access",
    fontsize=13, fontweight="bold"
)
ax.legend(title="Cluster", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"\n✅  Cluster plot saved to '{OUTPUT_PLOT}'")


# ── Save ──────────────────────────────────────────────────────────────────────

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅  Cluster assignments saved to '{OUTPUT_CSV}'")
