import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a-b)**2))

    def fit(self, X, max_iters=100):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size= (self.k, X.shape[1]))
        for _ in range(max_iters):
            y=[]

            for data_point in X:
                distances = [self.euclidean_distance(data_point, centroid) for centroid in self.centroids]
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i).flatten())

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])  # keep the old centroid if no points are assigned
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))
                
            if np.max(np.abs(self.centroids - cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y
    

from sklearn.datasets import make_blobs

# generates data with real, obvious clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=8, random_state=42)

kmeans = KMeansClustering(k=3)
clusters = kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='*', s=200)
plt.show()