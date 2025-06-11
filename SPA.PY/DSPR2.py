import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df=pd.read_csv('DS_MajorPROJ2/sports_performance_data.csv')
print(df.head())

# Data Preprocessing
# Check for missing values
'''print(df.isnull().sum())
# Check for duplicate rows
print(df.duplicated().sum())
#describe the dataset
print(df.describe())

#EDA: Exploratory Data Analysis
# Visualize team vs success rate
plt.figure(figsize=(8, 8))
team_success = df.groupby('team_name')['success_rate'].mean()
plt.pie(team_success, labels=team_success.index, autopct='%1.1f%%', startangle=140)
plt.title('Success Rate Distribution by Team')
plt.tight_layout()
plt.show()

#Visualize sport distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sport', order=df['sport'].value_counts().index)
plt.title('Distribution of Sports')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize team_name vs rating using violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='team_name', y='rating', palette='viridis')
plt.title('Team Ratings Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Visualize team_name with their matches_won box plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='team_name', y='matches_won', palette='Set2')
plt.title('Matches Won by Team')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize team_name with their matches_lost line plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='team_name', y='matches_lost', marker='o', palette='Set1')
plt.title('Matches Lost by Team')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize heatmap of correlation of numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

#•	Perform regression/clustering to find patterns.
# Clustering using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
# Selecting features for clustering
features = df[['success_rate', 'rating', 'matches_won', 'matches_lost']]
kmeans.fit(features)
# Adding cluster labels to the original dataframe
df['cluster'] = kmeans.labels_
# Visualizing clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='success_rate', y='rating', hue='cluster', palette='Set1', s=100)
# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='black', marker='X', label='Centroids')
plt.title('KMeans Clustering of Teams')
plt.xlabel('Success Rate')
plt.ylabel('Rating')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
'''
#perform regression analysis
from sklearn.linear_model import LinearRegression
# Define features and target variable
# Use all numeric columns except 'speed' as feature variables
X = df.select_dtypes(include=[np.number]).drop(columns=['speed'])
y = df['speed']  # Use 'speed' as the target variable for regression

# Fit the regression model
model = LinearRegression()
model.fit(X, y)
# Print coefficients
print("Coefficients:", model.coef_)
# Print intercept
print("Intercept:", model.intercept_)
# Predicting speed based on features
predictions = model.predict(X)
# Display first 10 predictions
print("Predicted Speed:", predictions[:10])
# Visualize actual vs predicted speed
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.title('Actual vs Predicted Speed')
plt.xlabel('Actual Speed')
plt.ylabel('Predicted Speed')
plt.legend()
plt.tight_layout()
plt.show()
# Save the modified DataFrame with clusters to a new CSV file
df.to_csv('DS_MajorPROJ2/sports_performance_with_clusters.csv', index=False)


















