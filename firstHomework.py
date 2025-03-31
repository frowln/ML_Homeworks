import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import itertools
import time
import os

def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear')

class IrisKMeans:
    def __init__(self, random_state=42):
        self.data = load_iris()
        self.X = self.data.data
        self.feature_names = self.data.feature_names
        self.random_state = random_state
        self.history = []
        self.centroids = None
        self.labels = None

    def find_optimal_clusters(self, max_k=5):
        inertias = []
        silhouettes = []
        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state).fit(self.X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.X, kmeans.labels_))
        self.optimal_k = 3  # Автоматизацию можно добавить здесь
        return self.optimal_k

    def custom_kmeans(self, k, max_iters=100):
        np.random.seed(self.random_state)
        self.centroids = self.X[np.random.choice(self.X.shape[0], k, replace=False)]
        for _ in range(max_iters):
            distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            self.history.append((self.centroids.copy(), self.labels.copy()))
            new_centroids = []
            for i in range(k):
                cluster_points = self.X[self.labels == i]
                new_centroids.append(cluster_points.mean(axis=0) if len(cluster_points) > 0 else self.centroids[i])
            new_centroids = np.array(new_centroids)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.labels, self.centroids

    def visualize_steps(self):
        for step, (centroids, labels) in enumerate(self.history):
            clear_output()
            plt.figure(figsize=(6, 4))
            plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
            plt.title(f'Step {step+1}')
            plt.show()
            time.sleep(1)

    def plot_final_clusters(self):
        combs = list(itertools.combinations(range(4), 2))
        plt.figure(figsize=(15, 10))
        for i, (f1, f2) in enumerate(combs, 1):
            plt.subplot(2, 3, i)
            plt.scatter(self.X[:, f1], self.X[:, f2], c=self.labels, cmap='viridis')
            plt.scatter(self.centroids[:, f1], self.centroids[:, f2], c='red', marker='x', s=100)
            plt.xlabel(self.feature_names[f1])
            plt.ylabel(self.feature_names[f2])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ikm = IrisKMeans()
    optimal_k = ikm.find_optimal_clusters()
    ikm.custom_kmeans(optimal_k)
    ikm.visualize_steps()
    ikm.plot_final_clusters()