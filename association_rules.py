"""
Part 2 – Association Rule Discovery
Finds patterns between sports-infrastructure access and athletic performance.

Uses a manual implementation of Apriori since mlxtend may not be installed.

Input : school_clusters.csv  (from cluster.py)
Output: association_rules.csv
        rules_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

INPUT_FILE  = "school_clusters.csv"
RULES_FILE  = "association_rules.csv"
PLOT_FILE   = "rules_plot.png"

MIN_SUPPORT    = 0.2    # item appears in ≥ 20% of schools
MIN_CONFIDENCE = 0.5    # rule is correct ≥ 50% of the time
MIN_LIFT       = 1.0    # association is above chance


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} schools with columns: {list(df.columns)}")
n = len(df)


# ── Build transaction items per school ────────────────────────────────────────
# Each school is described by binary items derived from its features.
# Thresholds use the dataset median so items are balanced.

med_facilities = df["n_facilities"].median()
med_acres      = df["total_acres"].median()
med_athletes   = df["athletes"].median()
med_part       = df["participation_pct"].median()

def build_items(row):
    items = set()
    # Cluster label (e.g. "High Access Zone")
    items.add(f"CLUSTER:{row['cluster_label'].replace(' ', '_')}")
    # Facility access
    if row["n_facilities"] >= med_facilities:
        items.add("HIGH_FACILITIES")
    else:
        items.add("LOW_FACILITIES")
    # Green space
    if row["total_acres"] >= med_acres:
        items.add("HIGH_ACRES")
    else:
        items.add("LOW_ACRES")
    # Athletic participation
    if row["participation_pct"] >= med_part:
        items.add("HIGH_PARTICIPATION")
    else:
        items.add("LOW_PARTICIPATION")
    # Athlete count
    if row["athletes"] >= med_athletes:
        items.add("HIGH_ATHLETES")
    else:
        items.add("LOW_ATHLETES")
    # Competitive success
    if row["total_titles"] > 0:
        items.add("HAS_TITLES")
    else:
        items.add("NO_TITLES")
    if row["state_titles"] > 0:
        items.add("HAS_STATE_TITLES")
    if row["regional_titles"] >= 10:
        items.add("MANY_REGIONAL_TITLES")
    return frozenset(items)

df["items"] = df.apply(build_items, axis=1)
transactions = df["items"].tolist()

all_items = sorted(set(item for t in transactions for item in t))
print(f"\nTotal unique items: {len(all_items)}")
print("Items:", all_items)


# ── Manual Apriori ────────────────────────────────────────────────────────────

def support(itemset, transactions):
    return sum(itemset.issubset(t) for t in transactions) / len(transactions)

# Frequent 1-itemsets
freq1 = {frozenset([item]): support(frozenset([item]), transactions)
         for item in all_items}
freq1 = {k: v for k, v in freq1.items() if v >= MIN_SUPPORT}

# Frequent 2-itemsets
candidates2 = [frozenset(pair) for pair in combinations(
    sorted(set(item for k in freq1 for item in k)), 2
)]
freq2 = {c: support(c, transactions) for c in candidates2}
freq2 = {k: v for k, v in freq2.items() if v >= MIN_SUPPORT}

frequent_itemsets = {**freq1, **freq2}
print(f"\nFrequent itemsets (support ≥ {MIN_SUPPORT}): {len(frequent_itemsets)}")


# ── Generate association rules ────────────────────────────────────────────────

rules = []
for itemset, supp in freq2.items():
    items_list = list(itemset)
    for i in range(1, len(items_list)):
        for ant in combinations(items_list, i):
            ant_set = frozenset(ant)
            con_set = itemset - ant_set
            if not con_set:
                continue
            ant_supp = support(ant_set, transactions)
            if ant_supp == 0:
                continue
            conf = supp / ant_supp
            # Expected support if independent
            con_supp = support(con_set, transactions)
            lift = conf / con_supp if con_supp > 0 else 0
            if conf >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({
                    "antecedent":  " + ".join(sorted(ant_set)),
                    "consequent":  " + ".join(sorted(con_set)),
                    "support":     round(supp, 3),
                    "confidence":  round(conf, 3),
                    "lift":        round(lift, 3),
                })

rules_df = pd.DataFrame(rules).drop_duplicates()
rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"\nAssociation rules (conf ≥ {MIN_CONFIDENCE}, lift ≥ {MIN_LIFT}): {len(rules_df)}")
if len(rules_df):
    print(rules_df[["antecedent", "consequent", "support", "confidence", "lift"]]
          .head(15).to_string(index=False))


# ── Visualise ─────────────────────────────────────────────────────────────────

if len(rules_df) > 0:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        rules_df["support"], rules_df["confidence"],
        c=rules_df["lift"], cmap="YlOrRd",
        s=100, edgecolors="grey", linewidths=0.5, alpha=0.85
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Lift", fontsize=11)

    # Label the top-5 rules by lift
    for _, row in rules_df.head(5).iterrows():
        label = f"{row['antecedent']} → {row['consequent']}"
        ax.annotate(label, (row["support"], row["confidence"]),
                    fontsize=6.5, alpha=0.8,
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Support", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title(
        "Association Rules: Sports Infrastructure → Athletic Performance",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\n✅  Rules plot saved to '{PLOT_FILE}'")
else:
    print("\n⚠ No rules generated — try lowering MIN_SUPPORT or MIN_CONFIDENCE.")


# ── Save ──────────────────────────────────────────────────────────────────────

rules_df.to_csv(RULES_FILE, index=False)
print(f"✅  Rules saved to '{RULES_FILE}'")
