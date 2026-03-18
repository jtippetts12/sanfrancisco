# San Francisco.
Data Mining course project based on the data sets from https://data.sfgov.org/ 

# Assignment 1: San Francisco
### Sports Infrastructure, School Athletics & Neighborhood Safety
**Jason — Data Mining — Spring 2026**
**This project involved prompts to Gemini and Claude AI. Responses helped me understand concepts, troublshoot problems, and overall achieve more work than I could have other wise.
**

---

## 1. Introduction

This project applies data mining techniques to help a prospective San Francisco resident make an informed decision about where to live — particularly if they have school-age children with a serious interest in competitive athletics. The central question is whether a neighborhood's geographic access to city-funded parks and recreational facilities is a meaningful predictor of its high schools' competitive athletic performance, and how crime levels layer on top of that picture.

Three publicly available datasets were collected and combined: the SF Recreation and Parks Properties dataset, the SF Public Schools dataset (both from [SF Open Data](https://data.sfgov.org)), and a self-collected dataset of CIF-SF section championship results for SFUSD high schools. A fourth dataset — SFPD incident reports from May 2025 to present — was later incorporated to extend the analysis to neighborhood-level safety.

The project applies three data mining techniques: **K-Means clustering** (applied twice — once to schools, once to neighborhoods), **association rule discovery** using a custom Apriori implementation, and a combined **geospatial analysis** visualized as an interactive choropleth map.

---

## 2. Deployment Architecture

### Data Sources

| File | Source | Contents |
|---|---|---|
| `schools.csv` | [SF Open Data](https://data.sfgov.org/Economy-and-Community/SF-Public-Schools/ttfe-5cpq) | 235 schools with coordinates, grade ranges, neighborhoods |
| `facilities.csv` | [SF Open Data](https://data.sfgov.org/Culture-and-Recreation/Recreation-and-Parks-Properties/wkn6-jn8k) | 245 Recreation & Parks properties with acreage and location |
| `performance.csv` | Self-collected ([CIF-SF](http://www.cifsf.org/)) | 13 SFUSD high schools: state/regional titles, athletes, participation % |
| `police.csv` | [SF Open Data / SFPD](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783) | 79,107 incident reports from May 2025 to present |

### Pipeline

```
schools.csv + facilities.csv + performance.csv
              |
        preprocess.py  -->  school_features.csv
              |
          cluster.py   -->  school_clusters.csv + cluster_plot.png
              |
   association_rules.py --> association_rules.csv + rules_plot.png

facilities.csv + police.csv
              |
  neighborhood_cluster.py --> neighborhood_clusters.csv + neighborhood_cluster_plot.png + neighborhood_heatmap.png

school_clusters.csv + neighborhood_clusters.csv
              |
        build_map.py  -->  sf_map.html
```

**Script descriptions:**

- **`preprocess.py`** — Filters schools to high schools (grades 9–12 and 8–12). Performs a vectorized haversine spatial join counting Recreation & Parks properties within 1 mile of each school and summing their acreage. Merges with performance data using fuzzy name matching.
- **`cluster.py`** — Standard-scales features, selects optimal k via silhouette score, runs K-Means, assigns descriptive cluster labels.
- **`association_rules.py`** — Builds a transaction table encoding each school as binary items (cluster label, facility access, athlete count, title presence). Mines frequent itemsets and generates association rules via a custom Apriori implementation.
- **`neighborhood_cluster.py`** — Aggregates park infrastructure and crime incidents per neighborhood. Applies log-scaling to crime counts before clustering to prevent extreme outliers from dominating. Outputs cluster assignments and two visualizations.
- **`build_map.py`** — Combines both cluster outputs into a single self-contained `sf_map.html` using Leaflet.js with embedded data and an interactive hover panel.

### Setup & Running

```bash
pip install pandas numpy scikit-learn matplotlib
```

```bash
python preprocess.py
python cluster.py
python association_rules.py
python neighborhood_cluster.py
python build_map.py
```

Open `sf_map.html` in any browser — no server required.

---

## 3. Key Moments of Learning

### 3.1 Data Cleaning: Fuzzy Name Matching Across Sources

The performance dataset was self-collected from the CIF-SF website while the schools dataset came from SF Open Data. Names didn't match exactly across the two sources, and two schools (Marshall and SF International) used `low_grade = 8` rather than 9, so the initial high school filter silently dropped them. This was an early lesson about the gap between clean textbook data and real open data: minor inconsistencies in naming conventions and grade-range coding can cause silent data loss. The fix required a fuzzy matching function and widening the grade filter — but the more important takeaway was the value of explicitly reporting unmatched records rather than silently dropping them.

### 3.2 Clustering: Why Raw Crime Counts Needed Log-Scaling

The first neighborhood clustering attempt used raw crime counts and returned k=2: Tenderloin/Mission/SoMa on one side, every other neighborhood on the other. This was technically correct — those areas have 10–15× the incident volume of most of SF — but analytically useless for distinguishing among the other 37 neighborhoods. Applying `log1p()` to crime features before scaling compressed the extreme outliers while preserving relative ordering, producing k=4 with four genuinely informative clusters. Knowing *when* to transform features is as important as knowing which algorithm to apply.

### 3.3 Association Rules: Small Datasets Require Careful Threshold Tuning

With only 13 schools, every percentage point of support represents less than one school. Setting `MIN_SUPPORT` too high returned zero rules; too low returned trivially obvious ones. Finding a meaningful middle ground (support ≥ 0.20, confidence ≥ 0.50, lift ≥ 1.0) required iterating on thresholds and manually inspecting results. This reinforced the course guidance on avoiding false discoveries: the rules here are directionally interesting but would require a larger multi-year dataset to confirm statistically.

### 3.4 Geospatial Work: The Haversine Formula

Linking parks to schools required computing distances between lat/lon pairs. Naive Euclidean distance on raw coordinates is inaccurate at SF's latitude (~37.7°N), where one degree of longitude is only ~53 miles rather than the ~69 miles of a degree of latitude. The haversine formula correctly handles this by computing great-circle distance on the Earth's surface. Vectorizing with NumPy arrays made the spatial join across 13 schools × 241 facilities run in milliseconds.

---

## 4. Findings

### 4.1 School Athletic Clustering (k=4, silhouette=0.535)

| Cluster | Schools | Avg. Facilities | Avg. Acres | Avg. Titles |
|---|---|---|---|---|
| High Access Zone | McAteer, Mission High, O'Connell | 27 | 133 | 7 |
| Mid Access Zone | Galileo, Lincoln, Marshall, Wallenberg, Balboa, SF International | 18 | 133 | 16 |
| Low Access Zone | Lowell, Washington | 12 | 756 | 194 |
| Zone 4 | Burton, Jordan | 10 | 373 | 0 |

The **Low Access Zone** schools — Lowell (339 total titles) and Washington (49) — sit near large open spaces rather than dense park networks, yet dominate competitive athletics. This challenges the initial hypothesis that more nearby facilities drives better athletic outcomes.

### 4.2 Association Rule Findings

The strongest rules (lift > 1.8):

| Rule | Confidence | Lift | Interpretation |
|---|---|---|---|
| State titles → Many regional titles | 0.83 | 2.17 | Winning at state level strongly predicts regional success too |
| Many regional titles → State titles | 1.00 | 2.17 | Every school with many regional titles also has a state title |
| High athletes → State titles | 0.86 | 1.86 | Larger programs are disproportionately likely to win at state level |
| No titles → High acreage | 1.00 | 1.86 | Every school with no titles is near above-median green space |
| Low facilities → High athletes | 0.83 | 1.55 | Schools in lower-facility areas tend to have more athletes |

The overall pattern: established programme size is a stronger predictor of competitive success than nearby park infrastructure. The "more parks = better athletes" hypothesis is not supported by the data.

### 4.3 Neighborhood Safety Clustering (k=4, silhouette=0.42)

| Cluster | Avg. Incidents | Avg. Acres | Neighborhoods |
|---|---|---|---|
| High Crime | 7,994 | 32 | Tenderloin, Mission, SoMa, Bayview, Financial District |
| Moderate — Higher Crime | 1,183 | 91 | Most of central/western SF (28 neighborhoods) |
| Green & Moderate Crime | 913 | 965 | Golden Gate Park, Lakeshore |
| Safest (Less Green) | 93 | 183 | Presidio, Twin Peaks, Seacliff, Lincoln Park, McLaren Park |

The High Crime neighborhoods have the *least* green space (avg. 32 acres). The safest neighborhoods are low-density residential and government-managed areas on the city's western and southern edges. This challenges a simple "more parks = safer" narrative and points to population density as the underlying driver.

For a new resident: the **Safest** cluster offers the lowest crime by a wide margin, while the **Green & Moderate Crime** cluster (Lakeshore in particular) offers the best combination of green space and manageable crime — and is also home to Lowell High, the most decorated athletic program in the dataset.

### 4.4 Combined Map Observations

The interactive map (`sf_map.html`) overlays both analyses. Key spatial patterns:

- **Lowell High** (Low Access Zone, 339 titles, 505 athletes) sits in a Safest/Green neighborhood — the single school that scores well on both safety and athletic output.
- The three **High Access Zone** schools (Mission, O'Connell, McAteer) are all in or adjacent to High Crime neighborhoods. High park density and high crime can coexist.
- **McAteer** has the highest participation rate in the dataset (57.4%) despite zero competitive titles — broad engagement with sport without championship-level results.

---

## 5. Limitations

- Performance data covers only 13 of 33 SF high schools; findings should not be generalized to the full district.
- CIF-SF records are not broken down by sport, so sport-specific patterns could not be explored.
- Crime data covers approximately ten months and may not reflect long-term trends.
- Neighborhood polygons in the map are simplified approximations, not official GIS boundaries.

---

## 6. Data Sources

- **SF Public Schools:** https://data.sfgov.org/Economy-and-Community/SF-Public-Schools/ttfe-5cpq
- **Recreation & Parks Properties:** https://data.sfgov.org/Culture-and-Recreation/Recreation-and-Parks-Properties/wkn6-jn8k
- **SFPD Incident Reports:** https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783
- **CIF-SF Championship Archive:** http://www.cifsf.org/