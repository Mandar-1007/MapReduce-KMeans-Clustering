# MapReduce-KMeans-Clustering
This project implements the K-means clustering algorithm using Java and Hadoop MapReduce. It processes a large dataset of 3-dimensional points to iteratively assign points to clusters and update cluster centroids until convergence. The project demonstrates big data processing techniques, distributed computation, and scalable clustering using multiple iterations.

# Key features:
1) Multi-iteration K-means implementation.
2) Customizable number of clusters (K) and iterations (R)
3) Developed using IntelliJ IDEA, designed for Hadoop-compatible environments

# Dataset Creation
1) Points Dataset:
    - 5,000+ 3-dimensional points (x, y, z), where: x ranges from 0 to 10,000.
    - y and z are randomly generated within a defined range.
2) Seed Points:
    - A file containing K randomly chosen seed points, where K is a configurable parameter.
3) Output Files:
    - 3d_points_dataset.csv: Dataset of 3D points.
    - seed_points_K.csv: Initial cluster centers.
  
# K-means Variants
This project implements and compares four variations of K-means clustering using Hadoop MapReduce:

Task 1: Single-Iteration K-means (R=1):
Executes one iteration of the K-means algorithm to assign points to clusters and compute new centers.

Task 2: Basic Multi-Iteration K-means (R=5):
Executes the K-means algorithm for a fixed number of iterations (R=5), without checking for early convergence.

Task 3: Advanced Multi-Iteration K-means with Early Termination:
Includes an early termination condition: Stops if cluster centers remain unchanged over two consecutive iterations or meet a predefined threshold.

Task 4: Optimized Multi-Iteration K-means:
Introduces Hadoop optimizations: 1) Uses a combiner to reduce intermediate data size 2) Improves Mapper and Reducer logic for faster convergence

# Output Variations
The project produces two types of outputs:
1) Cluster Centers:
Final cluster centers with a flag indicating whether convergence was reached.
2) Clustered Data:
The dataset with points labeled by their assigned cluster centers.

# Clustering Evaluation
Evaluation Metric:
1) Silhouette Score

Measures the quality of clustering by comparing intra-cluster cohesion and inter-cluster separation.

Implemented using MapReduce
- Mapper: Computes distances between points and clusters.
- Reducer: Aggregates results to compute the Silhouette score for each cluster and overall dataset.

**Purpose:**
Evaluate and compare the performance of different K-means variations

# Installation and Usage
**Prerequisites:**
1) Java Development Kit (JDK) 8 or higher.
2) Apache Hadoop installed and configured.
3) IntelliJ IDEA (or any preferred IDE).
