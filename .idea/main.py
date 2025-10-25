# ================================
# Imports
# ================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.signal import detrend
import kagglehub

# ================================
# Downloading PulseDB
# ================================

path = kagglehub.dataset_download("weinanwangrutgers/pulsedb-balanced-training-and-testing")
# This is the database that I have found for this project

# ================================
# Configuration
# ================================

KAGGLE_DATA_PATH = path # Path to local Kaggle dataset download
FS = 100 # Sampling frequency (Hz)
SEGMENT_LENGTH = FS * 10 # 10-second segments

# ================================
# 1. Load PulseDB Segments
# ================================

def load_pulsedb_segments(path=KAGGLE_DATA_PATH, signal_type='ABP', num_segments=1000):
    segments = []
    for file in os.listdir(path):

        if signal_type.lower() in file.lower() and (file.endswith('.npy') or file.endswith('.csv')):
            data = np.load(os.path.join(path, file)) if file.endswith('.npy') else np.loadtxt(os.path.join(path, file), delimiter=',')
            data = data[:SEGMENT_LENGTH]
            data = detrend(data)
            segments.append(data)

            if len(segments) >= num_segments:
                break

    if len(segments) == 0:
        print(f"No {signal_type} data found in {path}. Generating synthetic data...")
        # If no segments are found, segmants will be randomly created instead
        segments = [np.sin(np.linspace(0, 10*np.pi, SEGMENT_LENGTH)) + 0.1*np.random.randn(SEGMENT_LENGTH) for _ in range(num_segments)]
    return np.array(segments)


# ================================
# 2. Divide-and-Conquer Clustering
# ================================

def similarity(a, b):
    corr = np.corrcoef(a, b)[0, 1]
    return 1 - corr # distance = 1 - correlation

def recursive_cluster(segments, threshold=0.3):
    n = len(segments)
    if n <= 2:
        return [segments]

    dist_matrix = squareform(pdist(segments, metric=lambda u, v: similarity(u, v)))
    avg_dist = np.mean(dist_matrix)

    if avg_dist < threshold:
        return [segments]

    midpoint = n // 2
    left = recursive_cluster(segments[:midpoint], threshold)
    right = recursive_cluster(segments[midpoint:], threshold)
    return left + right

# ================================
# 3. Closest Pair Algorithm
# ================================

def closest_pair(segments):
    n = len(segments)
    if n < 2:
        return None, None, float('inf')
    best_i, best_j, best_dist = None, None, float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dist = similarity(segments[i], segments[j])
            if dist < best_dist:
                best_i, best_j, best_dist = i, j, dist
    return best_i, best_j, best_dist

# ================================
# 4. Maximum Subarray (Kadane’s Algorithm)
# ================================

def max_subarray(signal):
    max_sum = float('-inf')
    current_sum = 0
    start = end = s = 0
    for i in range(len(signal)):
        current_sum += signal[i]
        if current_sum > max_sum:
            max_sum = current_sum
            start, end = s, i
        if current_sum < 0:
            current_sum = 0
            s = i + 1
    return start, end, max_sum

# ================================
# 5. Visualization
# ================================

def plot_cluster_example(cluster, cluster_id):
    plt.figure(figsize=(10, 4))

    for seg in cluster:
        plt.plot(seg, alpha=0.4)

    plt.title(f'Cluster {cluster_id} — {len(cluster)} segments')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    # Since we are using plt.show, we need to exit out of the window to end the running of the code

# ================================
# Main Pipeline
# ================================

def main():
    segments = load_pulsedb_segments()
    clusters = recursive_cluster(segments)
    print(f"Generated {len(clusters)} clusters.")

    for i, cluster in enumerate(clusters[:5]):
        print(f'Cluster {i+1}: {len(cluster)} segments')
        i1, i2, dist = closest_pair(cluster)

        if i1 is not None:
            print(f' Closest pair distance: {dist:.4f}')

        start, end, max_sum = max_subarray(cluster[0])
        print(f' Max activity interval: {start}-{end}, sum = {max_sum:.2f}')
        plot_cluster_example(cluster, i + 1)

if __name__ == '__main__':
    main()
