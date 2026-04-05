# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# -------------------------------
# 1. Data Collection and Analysis
# -------------------------------

# Loading data from CSV file
customer_data = pd.read_csv('Mall_Customers.csv')

# Display the first 5 rows of the dataset
customer_data.head()

# Display number of rows and columns
customer_data.shape

# Display data types and non-null information
customer_data.info()

# Check for missing values in each column
customer_data.isnull().sum()

# Selecting the features for clustering:
# Annual Income (column 3) and Spending Score (column 4)
X = customer_data.iloc[:, [3, 4]].values
print(X)

# -------------------------------
# 2. Finding Optimal Clusters (Elbow Method)
# -------------------------------

wcss = []  # List to store Within-Cluster Sum of Squares values

# Trying 1 to 10 clusters to see which one works best
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ gives WCSS

# Plotting the WCSS values to identify the elbow point
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# -------------------------------
# 3. Training K-Means Model with Optimal Cluster Count (k = 5)
# -------------------------------

# Creating and training K-Means with 5 clusters (based on elbow method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Printing the assigned cluster number for each customer
print(y_kmeans)

# -------------------------------
# 4. Visualizing the Clusters
# -------------------------------

plt.figure(figsize=(8, 8))

# Plotting cluster 1 (label 0)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s=100, c='red', label='Cluster 1')

# Plotting cluster 2 (label 1)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s=100, c='blue', label='Cluster 2')

# Plotting cluster 3 (label 2)
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], 
            s=100, c='green', label='Cluster 3')

# Plotting cluster 4 (label 3)
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], 
            s=100, c='cyan', label='Cluster 4')

# Plotting cluster 5 (label 4)
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], 
            s=100, c='magenta', label='Cluster 5')

# Plotting the centroids of each cluster
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids')

# Setting plot title and axis labels
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
