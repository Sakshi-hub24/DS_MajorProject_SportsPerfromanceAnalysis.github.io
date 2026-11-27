# Sports Performance Analysis

An end-to-end analytics project on **10,000 player records** across four sports. It combines Python (EDA, KMeans clustering, regression) with an interactive **Streamlit** app and a **Power BI** dashboard for exploring team and player performance.

![Streamlit demo](SPA%20Streamlit/SSR%202.gif)

## Dashboard preview

![Power BI dashboard](SPA%20Dashboard/dashboard.png)

## Highlights

- Cleaned and explored **10,000 player records** across 4 sports (Basketball, Cricket, Football, Hockey) and 4 teams.
- Created **8 EDA visualizations**, such as goals vs speed, success rate vs passes completed, and team wins and losses with and without injuries.
- Segmented players with **KMeans clustering** on performance metrics.
- Built a **multi-output Random Forest regression** to model `goals` and `success_rate`, with feature-importance plots.
- Built a **Streamlit** app with regression, clustering and top-performer views.
- Built a **Power BI** dashboard with team, player and sport slicers and KPI cards for assists, goals, minutes played and saves.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Power BI |
| Machine learning | scikit-learn (KMeans, RandomForestRegressor, MultiOutputRegressor) |
| App | Streamlit |

## Dataset

- Source: public Kaggle dataset, included in this repo as `sports_performance_data.csv`.
- 10,000 rows covering player and team stats such as `goals`, `assists`, `saves`, `passes_completed`, `minutes_played`, `speed`, `injuries`, `matches_played`, `matches_won`, `matches_lost`, `score` and `success_rate`, plus `sport`, `team_name` and `player_name`.
- The data appears to be synthetic: for example, the correlations between goals, speed, assists and success rate are close to zero. The project demonstrates the **analytics workflow and tooling**, not real sporting conclusions.

## Approach

1. **Clean and explore:** check missing values and duplicates, summarize the data, and visualize distributions and relationships.
2. **Cluster:** group players with KMeans, saved to `sports_performance_with_clusters.csv`.
3. **Model:** fit a multi-output Random Forest for `goals` and `success_rate` using assists, saves, passes completed, minutes played, injuries and speed.
4. **Interpret:** plot feature importances and list top performers.
5. **Present:** deliver the results in Streamlit and Power BI.

## Project structure

```
.
├── SPA.PY/
│   ├── DSPR2.py                          # EDA, clustering, regression
│   ├── DSstreamlit.py                    # Streamlit app
│   ├── sports_performance_with_clusters.csv
│   └── V1.png ... V8.png                 # EDA charts
├── SPA Streamlit/                        # demo recording
├── SPA Dashboard/
│   ├── Sports Performance.pbix
│   └── dashboard.png
├── SPA PPT/                              # project presentation (PDF and PPTX)
├── sports_performance_data.csv
└── README.md
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/Sakshi-hub24/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# 3. Launch the app
cd SPA.PY
streamlit run DSstreamlit.py
```

Make sure `sports_performance_data.csv` is in the same folder as the script (copy it from the repo root, or update the path in the code). To explore the dashboard, open `Sports Performance.pbix` in Power BI Desktop.

## Limitations

- The regression is currently fitted and scored on the same data, so the reported R² reflects training fit, not predictive performance. A proper train/test split shows that these features have little predictive power on this dataset.
- The data looks synthetic, so findings should not be read as real-world insights.
- Number of clusters and features for KMeans should be justified (for example with the elbow method or silhouette score).

## Future improvements

- Add a **train/test split with cross-validation** and compare against a baseline.
- Choose the number of clusters with the elbow method and silhouette score.
- Frame a classification problem (for example match win or loss) and report precision and recall.
- Test the workflow on a real-world sports dataset.

## Author

**Sakshi Ambali**: [LinkedIn](https://linkedin.com/in/sakshi-ambali-83b0873a4) | [GitHub](https://github.com/Sakshi-hub24)
