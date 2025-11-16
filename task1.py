import numpy as np
import pandas as pd
from collections import Counter

# -------------------------------
# Distance Functions
# -------------------------------

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def cosine(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - (num / (den + 1e-10))

def jaccard(a, b):
    # Works only for binary data; your MNIST-like data is 0/255 or 0/1
    a_bin = (a > 0).astype(int)
    b_bin = (b > 0).astype(int)
    intersection = np.sum(a_bin * b_bin)
    union = np.sum(np.maximum(a_bin, b_bin))
    if union == 0:
        return 1
    return 1 - intersection / union


# -------------------------------
# K-Means Implementation
# -------------------------------

class KMeansCustom:
    def __init__(self, K, distance="euclidean", max_iter=500):
        self.K = K
        self.max_iter = max_iter
        self.distance_type = distance

    def _dist(self, a, b):
        if self.distance_type == "euclidean":
            return euclidean(a, b)
        elif self.distance_type == "cosine":
            return cosine(a, b)
        elif self.distance_type == "jaccard":
            return jaccard(a, b)
        else:
            raise ValueError("Unknown distance function")

    def fit(self, X):
        n, d = X.shape

        # Random centroid initialization
        np.random.seed(42)
        self.centroids = X[np.random.choice(n, self.K, replace=False)]

        prev_sse = None

        for it in range(self.max_iter):

            # Assign clusters
            labels = []
            for i in range(n):
                dists = [self._dist(X[i], c) for c in self.centroids]
                labels.append(np.argmin(dists))

            labels = np.array(labels)

            # Update centroids
            new_centroids = []
            for k in range(self.K):
                pts = X[labels == k]
                if len(pts) == 0:
                    new_centroids.append(self.centroids[k])
                else:
                    new_centroids.append(np.mean(pts, axis=0))

            new_centroids = np.array(new_centroids)

            # Compute SSE
            sse = 0
            for i in range(n):
                c = labels[i]
                sse += self._dist(X[i], new_centroids[c]) ** 2

            # stopping conditions
            if prev_sse is not None and sse > prev_sse:
                break
            if np.allclose(self.centroids, new_centroids):
                break

            prev_sse = sse
            self.centroids = new_centroids

        self.labels_ = labels
        self.sse_ = prev_sse
        return self

    def majority_vote_accuracy(self, y):
        correct = 0
        for k in range(self.K):
            idx = np.where(self.labels_ == k)[0]
            if len(idx) == 0:
                continue
            cluster_labels = y[idx]
            maj = Counter(cluster_labels).most_common(1)[0][0]
            correct += np.sum(cluster_labels == maj)
        return correct / len(y)


# -------------------------------
# Run all distance metrics
# -------------------------------

def run_all(X, y, K):
    methods = ["euclidean", "cosine", "jaccard"]
    results = {}

    for m in methods:
        print(f"\n====== Running {m.upper()} ======")
        model = KMeansCustom(K=K, distance=m)
        model.fit(X)
        acc = model.majority_vote_accuracy(y)

        print(f"SSE = {model.sse_}")
        print(f"Accuracy = {acc}")

        results[m] = {
            "SSE": model.sse_,
            "Accuracy": acc
        }

    return results


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    print("Loading data...")

    X = pd.read_csv("kmeans_data/data.csv", header=None).values
    y = pd.read_csv("kmeans_data/label.csv", header=None).values.ravel()

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    K = len(np.unique(y))
    print("Number of classes (K):", K)

    results = run_all(X, y, K)

    print("\nFinal Results:")
    print(results)