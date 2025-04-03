import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
cars_to_buy = pd.read_csv("car_sales_data.csv")

# Explore the dataset
print(cars_to_buy.describe())
print(cars_to_buy.info())  # Check for non-numeric data

# Prepare data for clustering (encode categorical variables)
new_data = cars_to_buy.drop(columns=['Car_Model','Region','Defect_Rate'])
new_data = pd.get_dummies(new_data, drop_first=True)  # Convert categorical variables to numeric
print(new_data.head())

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(new_data)
cars_to_buy['kmeans_cluster'] = kmeans.labels_

# Elbow method to find optimal k
wss = []
for k in range(1, 16):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(new_data)
    wss.append(kmeans.inertia_)

plt.plot(range(1, 16), wss, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Total within-clusters sum of square')
plt.title('Elbow Method for Optimal k')
plt.show()

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(new_data)

# Reduce dimensions for visualization (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Visualize the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Silhouette Score for K-Means
silhouette_avg = silhouette_score(new_data, cars_to_buy['kmeans_cluster'])
print("Mean Silhouette Width for K-Means Clustering:", silhouette_avg)

# Hierarchical clustering
linkage_matrix = linkage(new_data, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
cars_to_buy['hierarchical_cluster'] = agglomerative.fit_predict(new_data)

# Plot Hierarchical clusters
plt.scatter(new_data.iloc[:, 0], new_data.iloc[:, 1], 
            c=cars_to_buy['hierarchical_cluster'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel(new_data.columns[0])
plt.ylabel(new_data.columns[1])
plt.title('Hierarchical Clustering')
plt.show()

# Silhouette Score for Hierarchical Clustering
silhouette_avg_hierarchical = silhouette_score(new_data, cars_to_buy['hierarchical_cluster'])
print("Mean Silhouette Width for Hierarchical Clustering:", silhouette_avg_hierarchical)

# --- Pattern Recognition and Feature Extraction --- 

# 1. Analyze the mean of each feature within each cluster
# Only select numeric columns (filter non-numeric)
numeric_columns = new_data.select_dtypes(include=[np.number]).columns

kmeans_cluster_means = cars_to_buy.groupby('kmeans_cluster')[numeric_columns].mean()
hierarchical_cluster_means = cars_to_buy.groupby('hierarchical_cluster')[numeric_columns].mean()

print("\nCluster characteristics (K-Means):")
print(kmeans_cluster_means)

print("\nCluster characteristics (Hierarchical):")
print(hierarchical_cluster_means)

# 2. Find the most important features for each cluster (high variance)
# We can compute the standard deviation or variance of features within each cluster
kmeans_cluster_variance = cars_to_buy.groupby('kmeans_cluster')[numeric_columns].std()
hierarchical_cluster_variance = cars_to_buy.groupby('hierarchical_cluster')[numeric_columns].std()

print("\nFeature Variance within K-Means clusters:")
print(kmeans_cluster_variance)

print("\nFeature Variance within Hierarchical clusters:")
print(hierarchical_cluster_variance)

# 3. Distribution of the clusters for each feature
# Plotting the distributions (boxplots) for key features
features_to_plot = ['Sales', 'Price']  # Add more features as needed

for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    plt.boxplot([cars_to_buy[cars_to_buy['kmeans_cluster'] == i][feature] for i in range(3)], 
                labels=[f'Cluster {i}' for i in range(3)])
    plt.title(f'{feature} Distribution by K-Means Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'{feature}')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.boxplot([cars_to_buy[cars_to_buy['hierarchical_cluster'] == i][feature] for i in range(3)], 
                labels=[f'Cluster {i}' for i in range(3)])
    plt.title(f'{feature} Distribution by Hierarchical Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'{feature}')
    plt.show()

# 4. Pattern Recognition - Detect if there are patterns across clusters
# Let's look at the distribution of car features across clusters using mean and variance

# Find the mean and variance for each cluster from K-Means clustering
kmeans_cluster_pattern = kmeans_cluster_means.transpose()
print("\nK-Means Cluster Patterns (Mean values of features):")
print(kmeans_cluster_pattern)

# Find the mean and variance for each cluster from Hierarchical clustering
hierarchical_cluster_pattern = hierarchical_cluster_means.transpose()
print("\nHierarchical Cluster Patterns (Mean values of features):")
print(hierarchical_cluster_pattern)

# --- Feature Importance (Optional: If you want to get importance of features for clustering) ---
# We can use PCA to understand how much each feature contributes to the components that separate clusters
pca_components = pca.components_
feature_importance = pd.DataFrame(pca_components, columns=new_data.columns, index=[f"PC{i+1}" for i in range(pca_components.shape[0])])
print("\nPCA Feature Importance:")
print(feature_importance)
