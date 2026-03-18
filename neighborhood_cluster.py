"""
Neighborhood Safety & Park Access Clustering
Clusters all 40 SF neighborhoods by park infrastructure and crime profile.
Uses log-scaling on crime counts so extreme outliers (Tenderloin, Mission, SoMa)
don't collapse all other neighborhoods into a single cluster.

Input : facilities.csv, police.csv
Output: neighborhood_clusters.csv
        neighborhood_cluster_plot.png
        neighborhood_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

FAC_FILE   = "data/facilities.csv"
CRIME_FILE = "data/police.csv"
OUT_CSV    = "neighborhood_clusters.csv"
OUT_PLOT   = "neighborhood_cluster_plot.png"
OUT_HEAT   = "neighborhood_heatmap.png"

VIOLENT  = {'Assault','Robbery','Homicide','Rape','Sex Offense',
            'Weapons Offense','Weapons Carrying Etc','Arson'}
PROPERTY = {'Larceny Theft','Burglary','Motor Vehicle Theft',
            'Stolen Property','Vandalism','Malicious Mischief'}
DRUG_DIS = {'Drug Offense','Drug Violation','Disorderly Conduct',
            'Civil Sidewalks','Prostitution'}

COLORS = ["#2ecc71","#e74c3c","#3498db","#f39c12","#9b59b6","#1abc9c","#e67e22"]

# ── Step 1: Facilities per neighborhood ───────────────────────────────────────

fac = pd.read_csv(FAC_FILE, encoding="utf-8-sig")
sf_fac = fac[fac["city"].str.lower().str.strip() == "san francisco"].copy()
sf_fac["acres"] = pd.to_numeric(sf_fac["acres"], errors="coerce").fillna(0)

rows = []
for _, r in sf_fac.iterrows():
    if pd.isna(r["analysis_neighborhood"]): continue
    for n in r["analysis_neighborhood"].split(","):
        n = n.strip()
        if n:
            rows.append({"neighborhood": n, "acres": r["acres"]})

fac_agg = pd.DataFrame(rows).groupby("neighborhood").agg(
    n_facilities=("acres","count"),
    total_acres =("acres","sum"),
).reset_index()

# ── Step 2: Crime per neighborhood ────────────────────────────────────────────

crime = pd.read_csv(CRIME_FILE, encoding="utf-8-sig")
crime["is_violent"]  = crime["Incident Category"].isin(VIOLENT).astype(int)
crime["is_property"] = crime["Incident Category"].isin(PROPERTY).astype(int)
crime["is_drug_dis"] = crime["Incident Category"].isin(DRUG_DIS).astype(int)

crime_agg = crime.groupby("Analysis Neighborhood").agg(
    total_incidents        =("is_violent","count"),
    violent_incidents      =("is_violent","sum"),
    property_incidents     =("is_property","sum"),
    drug_disorder_incidents=("is_drug_dis","sum"),
).reset_index().rename(columns={"Analysis Neighborhood":"neighborhood"})

# ── Step 3: Merge ─────────────────────────────────────────────────────────────

df = fac_agg.merge(crime_agg, on="neighborhood", how="inner")
print(f"Neighborhoods: {len(df)}")

# ── Step 4: Log-scale crime features, keep raw park features ─────────────────
# log1p compresses extreme outliers while preserving relative ordering

for col in ["total_incidents","violent_incidents","property_incidents","drug_disorder_incidents"]:
    df[f"log_{col}"] = np.log1p(df[col])

FEATURE_COLS = [
    "n_facilities", "total_acres",
    "log_total_incidents", "log_violent_incidents",
    "log_property_incidents", "log_drug_disorder_incidents",
]

X = df[FEATURE_COLS].fillna(0).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 5: K-Means with silhouette selection ─────────────────────────────────

print("Silhouette scores:")
best_k, best_score, best_model = 2, -1, None
for k in range(2, min(8, len(df))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  k={k}  silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score, best_model = k, score, km

print(f"\n→ Best k = {best_k}  (silhouette = {best_score:.3f})")
df["cluster"] = best_model.labels_

# ── Step 6: Label clusters ────────────────────────────────────────────────────

means = df.groupby("cluster")[["total_incidents","total_acres"]].mean()
crime_rank = means["total_incidents"].rank().astype(int)   # 1 = lowest crime
park_rank  = means["total_acres"].rank(ascending=False).astype(int)  # 1 = most green

def make_label(cid):
    cr = crime_rank[cid]   # 1=safest
    pr = park_rank[cid]    # 1=most green
    n  = best_k
    if cr == 1 and pr == 1:
        return "Safe & Green"
    elif cr == 1:
        return "Safest (Less Green)"
    elif cr == n and pr == 1:
        return "High Crime & Green"
    elif cr == n:
        return "High Crime"
    elif pr == 1:
        return "Green & Moderate Crime"
    elif cr <= n // 2:
        return "Moderate — Lower Crime"
    else:
        return "Moderate — Higher Crime"

df["cluster_label"] = df["cluster"].map(make_label)

print("\nCluster summary (raw values):")
raw_cols = ["n_facilities","total_acres","total_incidents","violent_incidents",
            "property_incidents","drug_disorder_incidents"]
print(df.groupby("cluster_label")[raw_cols].mean().round(1).to_string())

print("\nNeighborhoods per cluster:")
for label, group in df.groupby("cluster_label"):
    hoods = ", ".join(sorted(group["neighborhood"].tolist()))
    print(f"\n  [{label}]\n  {hoods}")

# ── Step 7: PCA scatter ───────────────────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca1"] = coords[:, 0]
df["pca2"] = coords[:, 1]

labels_ordered = (df.groupby("cluster_label")["total_incidents"]
                    .mean().sort_values().index.tolist())
color_map = {lab: COLORS[i % len(COLORS)] for i, lab in enumerate(labels_ordered)}

fig, ax = plt.subplots(figsize=(13, 9))
for label in labels_ordered:
    mask = df["cluster_label"] == label
    ax.scatter(
        df.loc[mask,"pca1"], df.loc[mask,"pca2"],
        color=color_map[label], s=140,
        edgecolors="white", linewidths=0.8,
        zorder=3, label=label
    )
for _, row in df.iterrows():
    ax.annotate(
        row["neighborhood"], (row["pca1"], row["pca2"]),
        fontsize=6.5, alpha=0.82,
        xytext=(5,3), textcoords="offset points"
    )

v1 = pca.explained_variance_ratio_[0]*100
v2 = pca.explained_variance_ratio_[1]*100
ax.set_xlabel(f"PC1  ({v1:.0f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2  ({v2:.0f}% variance)", fontsize=11)
ax.set_title("SF Neighborhood Clusters: Park Access vs. Crime Profile",
             fontsize=13, fontweight="bold")
ax.legend(title="Cluster", fontsize=9, title_fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"\n✅  Scatter plot → '{OUT_PLOT}'")

# ── Step 8: Feature heatmap ───────────────────────────────────────────────────

heat_cols = ["n_facilities","total_acres","total_incidents",
             "violent_incidents","property_incidents","drug_disorder_incidents"]
col_labels = ["Facilities\n(count)","Green Space\n(acres)","Total\nCrime",
              "Violent\nCrime","Property\nCrime","Drug/\nDisorder"]

df_sorted = df.sort_values(["cluster_label","total_incidents"])
heat_data = df_sorted.set_index("neighborhood")[heat_cols]
heat_norm = pd.DataFrame(
    MinMaxScaler().fit_transform(heat_data),
    index=heat_data.index, columns=heat_cols
)

fig2, ax2 = plt.subplots(figsize=(10, 14))
im = ax2.imshow(heat_norm.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

ax2.set_xticks(range(len(col_labels)))
ax2.set_xticklabels(col_labels, fontsize=9)
ax2.set_yticks(range(len(heat_norm)))
yticks = ax2.set_yticklabels(heat_norm.index, fontsize=8)
for tick, hood in zip(yticks, heat_norm.index):
    clabel = df_sorted.set_index("neighborhood").loc[hood,"cluster_label"]
    tick.set_color(color_map[clabel])

plt.colorbar(im, ax=ax2, label="Normalised value  (0 = lowest, 1 = highest)")
ax2.set_title("SF Neighborhoods: Park Access & Crime Profile\n(row labels coloured by cluster)",
              fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUT_HEAT, dpi=150)
print(f"✅  Heatmap → '{OUT_HEAT}'")

# ── Step 9: Save ──────────────────────────────────────────────────────────────
df[["neighborhood","cluster","cluster_label"] + raw_cols].to_csv(OUT_CSV, index=False)
print(f"✅  Data → '{OUT_CSV}'")
