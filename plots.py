import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ======================
# Task 1: K-Means  
# ======================

# Results from Task1
metrics_task1 = {
    "Euclidean": {"SSE": 25414767689.9611, "Accuracy": 0.5851},
    "Cosine":    {"SSE": 686.2753,           "Accuracy": 0.6264},
    "Jaccard":   {"SSE": 5387.1070,          "Accuracy": 0.1430}
}

# Plot SSE bar-chart
plt.figure(figsize=(8,5))
names = list(metrics_task1.keys())
sses = [metrics_task1[n]["SSE"] for n in names]
plt.bar(names, sses, color=['skyblue','salmon','lightgreen'])
plt.ylabel('SSE')
plt.title('Task1: SSE Comparison for Distance Metrics')
plt.yscale('log')  # log scale for readability
plt.tight_layout()
plt.savefig("task1_sse_comparison.png")
plt.show()

# Plot Accuracy bar-chart
plt.figure(figsize=(8,5))
accs = [metrics_task1[n]["Accuracy"] for n in names]
plt.bar(names, accs, color=['skyblue','salmon','lightgreen'])
plt.ylabel('Accuracy')
plt.title('Task1: Accuracy Comparison for Distance Metrics')
plt.tight_layout()
plt.savefig("task1_accuracy_comparison.png")
plt.show()


# ======================
# Task 2: Recommender Models  
# ======================

# Results from Task2
metrics_task2 = {
    "User-CF": {"MAE": 0.7674, "RMSE": 0.9940},
    "Item-CF": {"MAE": 0.7748, "RMSE": 0.9954},
    "PMF":     {"MAE": 0.6906, "RMSE": 0.8970}
}

# Plot MAE bar-chart
plt.figure(figsize=(8,5))
names2 = list(metrics_task2.keys())
maes = [metrics_task2[n]["MAE"] for n in names2]
plt.bar(names2, maes, color=['skyblue','salmon','lightgreen'])
plt.ylabel('MAE')
plt.title('Task2: MAE Comparison for Recommender Models')
plt.tight_layout()
plt.savefig("task2_mae_comparison.png")
plt.show()

# Plot RMSE bar-chart
plt.figure(figsize=(8,5))
rmses = [metrics_task2[n]["RMSE"] for n in names2]
plt.bar(names2, rmses, color=['skyblue','salmon','lightgreen'])
plt.ylabel('RMSE')
plt.title('Task2: RMSE Comparison for Recommender Models')
plt.tight_layout()
plt.savefig("task2_rmse_comparison.png")
plt.show()


# ======================
# Additional Plots (placeholders)
# ======================

# Similarity impact (You can fill with your own data)
plt.figure(figsize=(8,5))
# Suppose you collected results for different similarity metrics in User-CF
sim_methods = ['cosine','msd','pearson']
mae_sim = [0.7674, 0.7800, 0.7700]  # substitute actual values
plt.plot(sim_methods, mae_sim, marker='o')
plt.title('Task2: Impact of Similarity Metric on User-CF MAE')
plt.ylabel('MAE')
plt.xlabel('Similarity Metric')
plt.tight_layout()
plt.savefig("task2_similarity_impact.png")
plt.show()

# Neighbors impact (You can fill with your own data)
plt.figure(figsize=(8,5))
k_vals = [10,20,30,40,50]
rmse_k = [1.00,0.99,0.995,0.992,0.990]  # substitute actual values
plt.plot(k_vals, rmse_k, marker='o')
plt.title('Task2: Impact of Number of Neighbors on User-CF RMSE')
plt.ylabel('RMSE')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("task2_neighbors_impact.png")
plt.show()