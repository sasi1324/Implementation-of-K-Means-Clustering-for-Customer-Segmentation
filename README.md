# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sasinthara S
RegisterNumber:  212223110045
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female'],
    'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

df = pd.DataFrame(data)

df_numeric = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

inertia = []  

k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)  

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

optimal_k = 3  

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

print(df[['CustomerID', 'Cluster']])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

df_pca_df = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca_df['Cluster'] = df['Cluster']

plt.figure(figsize=(8, 6))
plt.scatter(df_pca_df[df_pca_df['Cluster'] == 0]['PCA1'], df_pca_df[df_pca_df['Cluster'] == 0]['PCA2'], s=100, c='red', label='Cluster 1')
plt.scatter(df_pca_df[df_pca_df['Cluster'] == 1]['PCA1'], df_pca_df[df_pca_df['Cluster'] == 1]['PCA2'], s=100, c='blue', label='Cluster 2')
plt.scatter(df_pca_df[df_pca_df['Cluster'] == 2]['PCA1'], df_pca_df[df_pca_df['Cluster'] == 2]['PCA2'], s=100, c='green', label='Cluster 3')

centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/04e8c1aa-1d1e-4693-bc93-e4a042513981)

![download (1)](https://github.com/user-attachments/assets/6e78f6e4-84d6-4ceb-8a4b-1da5fe3ec5fa)

![download](https://github.com/user-attachments/assets/e206baae-1a83-4cc6-9c2b-e7805c7e712b)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
