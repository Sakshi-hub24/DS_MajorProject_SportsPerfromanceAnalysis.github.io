import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page Setup
st.set_page_config(page_title="Sports Multi-Target Analyzer", layout="wide")
st.title("🏆 Sports Performance Multi-Target Analysis")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("sports_performance_data.csv")

df = load_data()
st.success("✅ Dataset Loaded")

# Sidebar Controls
st.sidebar.header("⚙️ Analysis Options")
view_data = st.sidebar.checkbox("Preview Dataset")
run_regression = st.sidebar.checkbox("Run Multi-Target Regression")
run_clustering = st.sidebar.checkbox("Run Clustering")
show_suggestions = st.sidebar.checkbox("Show Tactical Suggestions")

# Preview Dataset
if view_data:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20))

# Define Features & Targets
feature_cols = [
    'assists',
    'saves',
    'passes_completed',
    'minutes_played',
    'injuries',
    'speed'
]

target_cols = ['goals', 'success_rate']

X = df[feature_cols]
y = df[target_cols]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MULTI-TARGET REGRESSION
if run_regression:
    st.subheader("📈 Multi-Target Regression: Predicting Goals & Success Rate")

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y)

    r2_scores = model.score(X_scaled, y)
    st.markdown(f"**🎯 Overall R² Score:** `{r2_scores:.3f}`")

    # Feature Importances for each target
    importances_goals = model.estimators_[0].feature_importances_
    importances_success = model.estimators_[1].feature_importances_

    st.markdown("**🌲 Feature Importances for Goals Prediction:**")
    st.write(dict(zip(feature_cols, importances_goals.round(3))))

    st.markdown("**🌲 Feature Importances for Success Rate Prediction:**")
    st.write(dict(zip(feature_cols, importances_success.round(3))))

    # Visualization
    st.markdown("**📊 Goals vs Speed**")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x='speed', y='goals', hue='sport', alpha=0.6, ax=ax1)
    st.pyplot(fig1)

    st.markdown("**📊 Success Rate vs Passes Completed**")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='passes_completed', y='success_rate', hue='sport', alpha=0.6, ax=ax2)
    st.pyplot(fig2)

# CLUSTERING
if run_clustering:
    st.subheader("🔍 Clustering: Grouping Players by Performance")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='assists', y='goals', hue='cluster', palette='Set2', ax=ax3)
    ax3.set_title("Player Clusters Based on Assists & Goals")
    st.pyplot(fig3)

    st.dataframe(df[['player_name', 'team_name', 'goals', 'assists', 'cluster']].head(10))

# TACTICAL SUGGESTIONS
if show_suggestions:
    st.subheader("💡 Tactical Suggestions: High Goal Scorers & High Success Rate Players")
    top_goals = df[df['goals'] > df['goals'].quantile(0.90)]
    top_success = df[df['success_rate'] > df['success_rate'].quantile(0.90)]

    st.markdown("### Top 10 Goal Scorers")
    st.dataframe(top_goals[['player_name', 'team_name', 'goals', 'assists', 'speed']].sort_values('goals', ascending=False).head(10))

    st.markdown("### Top 10 Players with Highest Success Rate")
    st.dataframe(top_success[['player_name', 'team_name', 'success_rate', 'passes_completed', 'speed']].sort_values('success_rate', ascending=False).head(10))
