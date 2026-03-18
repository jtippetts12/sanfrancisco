"""
Part 2 – Preprocessing Pipeline
SF Sports Infrastructure vs. High School Athletic Performance

Data sources (local CSV files — place in the same directory as this script):
  - schools.csv      : SF Public Schools (SF Open Data)
  - facilities.csv   : Recreation & Parks Properties (SF Open Data)
  - performance.csv  : School regional/state results + athlete counts (self-collected)

Output: school_features.csv
"""

import re
import pandas as pd
import numpy as np
from math import radians, cos

# ── Config ────────────────────────────────────────────────────────────────────
RADIUS_MILES = 1.0          # search radius around each school
OUTPUT_FILE  = "school_features.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_array(slat, slon, lats, lons):
    """Vectorised great-circle distance in miles from one point to an array."""
    R = 3958.8
    slat_r = radians(slat)
    lats_r = np.radians(lats)
    a = (np.sin((lats_r - slat_r) / 2) ** 2
         + cos(slat_r) * np.cos(lats_r) * np.sin((np.radians(lons) - radians(slon)) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize(name):
    """Lowercase + strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


def fuzzy_match(perf_name, school_lookup):
    """
    Match a performance-file school name to a key in school_lookup.
    Strategy: exact → all-words-present → majority-words-present.
    """
    norm = normalize(perf_name)
    if norm in school_lookup:
        return norm
    words = set(norm.split())
    candidates = [k for k in school_lookup if words.issubset(set(k.split()))]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return max(candidates, key=lambda k: sum(c in k for c in norm))
    for key in school_lookup:
        if len(words & set(key.split())) >= max(1, len(words) - 1):
            return key
    return None


def safe_int(val):
    v = pd.to_numeric(val, errors="coerce")
    return 0 if pd.isna(v) else int(v)


def safe_float(val):
    v = pd.to_numeric(val, errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


# ── Step 1 – Load ─────────────────────────────────────────────────────────────

schools_df    = pd.read_csv("data/schools.csv",     encoding="utf-8-sig")
facilities_df = pd.read_csv("data/facilities.csv",  encoding="utf-8-sig")
perf_df       = pd.read_csv("data/performance.csv", encoding="utf-8-sig")

print(f"Loaded {len(schools_df)} schools | {len(facilities_df)} facilities | "
      f"{len(perf_df)} performance records")


# ── Step 2 – Filter to high schools ──────────────────────────────────────────
# Includes schools where high_grade == 12 and low_grade in [8, 9, 10]
# (Some SFUSD schools like Marshall and SF International start at grade 8)

hs = schools_df[
    (schools_df["high_grade"].astype(str) == "12") &
    (schools_df["low_grade"].astype(str).isin(["8", "9", "10"])) &
    (schools_df["latitude"].notna()) &
    (schools_df["longitude"].notna())
].copy()
print(f"High schools with coordinates: {len(hs)}")


# ── Step 3 – Match performance records to coordinates ────────────────────────

school_lookup = {normalize(r["school"]): r for _, r in hs.iterrows()}

match_results = []
for _, p in perf_df.iterrows():
    key = fuzzy_match(p["School"], school_lookup)
    if key:
        sch = school_lookup[key]
        match_results.append({
            "school_name":  sch["school"],
            "neighborhood": sch.get("analysis_neighborhood", ""),
            "latitude":     float(sch["latitude"]),
            "longitude":    float(sch["longitude"]),
            "State":        p.get("State"),
            "regional":     p.get("regional"),
            "athletes":     p.get("athletes"),
            "participation":p.get("participation"),
        })
    else:
        print(f"  ⚠ No coordinate match for: '{p['School']}'")

matched_df = pd.DataFrame(match_results)
print(f"Matched {len(matched_df)} / {len(perf_df)} performance schools")


# ── Step 4 – Filter facilities to SF ─────────────────────────────────────────

fac = facilities_df[
    (facilities_df["city"].str.lower().str.strip() == "san francisco") &
    (facilities_df["latitude"].notna()) &
    (facilities_df["longitude"].notna())
].copy()
fac["acres"] = pd.to_numeric(fac["acres"], errors="coerce").fillna(0)
fac_lats  = fac["latitude"].to_numpy(dtype=float)
fac_lons  = fac["longitude"].to_numpy(dtype=float)
fac_acres = fac["acres"].to_numpy(dtype=float)
print(f"SF facilities with coordinates: {len(fac)}")


# ── Step 5 – Spatial join ─────────────────────────────────────────────────────

rows = []
for _, s in matched_df.iterrows():
    slat, slon = s["latitude"], s["longitude"]
    mask = haversine_array(slat, slon, fac_lats, fac_lons) <= RADIUS_MILES
    rows.append({
        "school_name":       s["school_name"],
        "neighborhood":      s["neighborhood"],
        "latitude":          slat,
        "longitude":         slon,
        "n_facilities":      int(mask.sum()),
        "total_acres":       round(float(fac_acres[mask].sum()), 2),
        "state_titles":      safe_int(s["State"]),
        "regional_titles":   safe_int(s["regional"]),
        "athletes":          safe_int(s["athletes"]),
        "participation_pct": safe_float(s["participation"]),
    })

features_df = pd.DataFrame(rows)
features_df["total_titles"] = features_df["state_titles"] + features_df["regional_titles"]


# ── Step 6 – Save ─────────────────────────────────────────────────────────────

features_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFeature matrix ({len(features_df)} schools):")
print(features_df[[
    "school_name", "n_facilities", "total_acres",
    "state_titles", "regional_titles", "total_titles",
    "athletes", "participation_pct"
]].to_string(index=False))
print(f"\n✅  Saved to '{OUTPUT_FILE}'")
