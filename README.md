# DS_MajorProject_SportsPerfromanceAnalysis.github.io

# 🏆 Sports Performance Analysis

This project aims to analyze individual and team performance in various sports using data visualization and machine learning. It includes an interactive Streamlit app, a Power BI dashboard, and multiple Python scripts for regression, clustering, and KPI analysis.

---

## 📁 Project Structure
- `sports_performance_data.csv` – Main dataset containing 10,000+ player records.
- `sports_performance_with_clusters.csv` – Enhanced dataset with clustering results.
- `DSstreamlit.py` – Streamlit app for regression, clustering, and tactical suggestions.
- `DSPR2.py` – Python script for statistical plots, EDA, and regression visualization.
- `Power BI Dashboard` – Visual analysis with slicers, KPI cards, bar/line charts, and cluster summaries.

---

## 🎯 Objectives
- Analyze team and player performance using key stats.
- Predict multiple performance outcomes using machine learning.
- Visualize KPIs and trends through interactive dashboards.
- Suggest tactical improvements using clustering and regression.

---

## 🛠️ Tools & Technologies
- **Python**: Pandas, Matplotlib, Seaborn, Scikit-learn
- **Streamlit**: For creating an interactive web app
- **Power BI**: For building dynamic dashboards
- **Machine Learning**: RandomForestRegressor, MultiOutputRegressor, KMeans

---

## 📊 Key Features

### Streamlit App (`DSstreamlit.py`)
- 📌 Multi-target regression to predict both `goals` and `success_rate`
- 🔍 Feature importance plots for actionable insights
- 🔁 KMeans clustering to segment players by performance
- 🧠 Tactical suggestions for top players based on goals and efficiency
- 📈 Clean, responsive UI with compact charts

### Power BI Dashboard
- ✅ Filters by team, player, and sport
- 📊 KPI cards showing goals, assists, minutes played, and saves
- 📉 Analysis of injuries vs performance
- 📚 Clustering-based comparisons across sports

---

## 📁 How to Run

### 📌 Streamlit App
1. Install dependencies:
    pip install streamlit pandas seaborn matplotlib scikit-learn
2.Run the app:
    streamlit run DSstreamlit.py
3.Ensure the CSV file is in the same directory.

📌 Power BI Report
Open the .pbix file (or import the dataset into Power BI).
Use slicers to explore performance by team or player.

📈 Sample Visualizations
Goals vs Speed
Success Rate vs Passes Completed
Clustered Players by Goals and Assists
Team Wins/Losses with and without injuries
KPI Cards: Assists, Saves, Minutes Played

✅ Outcome
Built an end-to-end analytical tool combining Python + Power BI
Gained insights into what drives team and player performance
Delivered regression-based forecasts and clustering-based player evaluation
Created a user-friendly app usable by coaches, analysts, or sports managers

📬 Contact
Author: [Sakshi A]



