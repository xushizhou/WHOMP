# WHOMP: Wasserstein HOMogeneity Partition

## Overview

WHOMP (Wasserstein HOMogeneity Partition) is an open-source Python package designed to partition datasets into subgroups that maximize diversity within each subgroup while minimizing dissimilarity across subgroups. It optimally minimizes type I and type II errors that often result from imbalanced group splitting or partitioning, commonly referred to as accidental bias, in comparative and controlled trials.

## Usage

### Example: Balanced K-Means Clustering

This example demonstrates how to use WHOMP to perform balanced K-Means clustering, where the clusters have constraints on their sizes.

```python
from WHOMP import KMeansConstrained
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate toy data
X_toy, truth_model = make_blobs(n_samples=60, centers=[(0,10), (-10,-5), (10,-5)], cluster_std=[3, 3, 3])

# Apply KMeans with balanced clusters
clf = KMeansConstrained(n_clusters=3, size_min=20, size_max=20, random_state=0)
clf.fit_predict(X_toy)

# Plot the results
plt.scatter(X_toy[:, 0], X_toy[:, 1], c=clf.labels_, cmap='viridis', marker='o')
plt.title('Balanced K-Means Clustering')
plt.show()
```

### Example: WHOMP Random

WHOMP also provides functionality to generate evenly dispersed subgroups with ''anti-clustering'' techniques.

```python
from WHOMP import WHOMP_Random

# Apply WHOMP Random Anti-Clustering
anti_index = WHOMP_Random(X_toy, 2)

# Plot WHOMP Random Anti-Clustering result
for i in range(3):
    plt.scatter(X_toy[np.where(anti_index == i)[0], 0], X_toy[np.where(anti_index == i)[0], 1])
plt.title('WHOMP Random Anti-Clustering Result')
plt.show()
```

### Example: WHOMP Matching on a Toy Dataset

This example shows how to use WHOMP to perform barycenter matching, where the subgroups are matched to minimize the Wasserstein distance between them and the original dataset at the lowest scale or variance differences among the subgroups.

```python
from WHOMP import WHOMP_Matching

# Apply WHOMP Matching
barycenter, barysample_index = WHOMP_Matching(X_toy, 2, 0.0000001)

# Plot WHOMP Matching result
for i in range(2):
    plt.scatter(X_toy[barysample_index[i], 0], X_toy[barysample_index[i], 1])
plt.title('WHOMP Matching Result')
plt.show()
```

## Test Cases

WHOMP has several built-in test cases that evaluate the performance of its partitioning methods in various scenarios:

### 1. Wasserstein Experiment on Gaussian Dataset

This test demonstrates how to calculate Wasserstein-2 distances between subsampled groups and the original dataset using Gaussian blobs.

```python
from WHOMP_Test Wasserstein_Gaussian_experiment

random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave, random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std = Wasserstein_Gaussian_experiment()

print("\nAverage Errors:")
print("Random:", random_error_ave)
print("Pocock & Simon:", PS_error_ave)
print("Anticluster:", anticluster_error_ave)
print("WHOMP Barycenter Matching:", WHOMP_error_ave)

print("\nStandard Deviations:")
print("Random:", random_error_std)
print("Pocock & Simon:", PS_error_std)
print("Anticluster:", anticluster_error_std)
print("WHOMP Barycenter Matching:", WHOMP_error_std)
```

### 2. Wasserstein Experiment on NPI Dataset

This test case demonstrates how to apply WHOMP and Wasserstein distance calculations on an NPI dataset.

```python
import numpy as np 
import pandas as pd
from WHOMP_Test import NPI_experiment

# Load data, use your own path
df = pd.read_csv('/Users/shizhouxu/Desktop/SX_Workspace/Diverse_Subgroups/NPI/NPI_dataset/data.csv')

X = df.to_numpy()[:,1:41]
Gender = df['gender'].to_numpy()
Score = df['score'].to_numpy()
Gender = Gender[~np.any(X == 0, axis=1)]
Score = Score[~np.any(X == 0, axis=1)]
X = X[~np.any(X == 0, axis=1)]

# Run the partition experiment
random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave, \
random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std = NPI_experiment(X, range(2, 7, 2), 500)

# Print average and standard deviations of errors
print("Average Errors:", random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave)
print("Standard Deviations:", random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std)

# Print variance of partition results
for i in range(3):
    print(f"Variances (bary_var_list[{i}]):", np.var(np.array(bary_var_list[i])), np.var(np.array(anti_var_list[i])), np.var(np.array(PS_var_list[i])), np.var(np.array(random_var_list[i])))

for i in range(3):
    print(f"Mean Variances (bary_mean_list[{i}]):", np.var(np.array(bary_mean_list[i])), np.var(np.array(anti_mean_list[i])), np.var(np.array(PS_mean_list[i])), np.var(np.array(random_mean_list[i])))
```

### 3. Stochastic Block Model Spectrum Experiment

This experiment compares different subgroups by computing the Wasserstein distance between Laplacian spectra in a Stochastic Block Model (SBM). Different partitioning methods are used to generate subgroups.

```python
from WHOMP import SBM_spectrum_experiment

Ave_list, Std_list = [], []
for i in range(3):
    ave, std = SBM_spectrum_experiment((i+1)*2, 100)
    Ave_list.append(ave)
    Std_list.append(std)

# Print results
print("Averages: ", Ave_list)
print("Standard Deviations: ", Std_list)
```

### 4. Entropy on MNIST

This test evaluates the entropy of subgroups generated from MNIST t-SNE embeddings. The goal is to measure how well different partitioning methods preserve the original label distribution in each subgroup.

```python
from WHOMP import entropy_MNIST_experiment

repetition = 50
subgroup_number = 2
random_ent, PS_ent, anticluster_ent, WHOMP_ent = entropy_MNIST_experiment(tsne, subgroup_number, repetition)

# Print results
print("Random Entropy: ", np.mean(random_ent))
print("Pocock & Simon Entropy: ", np.mean(PS_ent))
print("WHOMP Random Entropy: ", np.mean(anticluster_ent))
print("WHOMP Matching Entropy: ", np.mean(WHOMP_ent))
```

## Dependencies

	•	numpy
	•	matplotlib
	•	scikit-learn
	•	joblib
	•	POT (Python Optimal Transport library)
	•	networkx
	•	scipy

## License

WHOMP is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

We welcome contributions! Please read our CONTRIBUTING.md for details on how to submit pull requests, report issues, and suggest features.

## Acknowledgements

WHOMP is built on the foundation of advanced partitioning techniques and optimal transport, and we thank the open-source community for providing the tools and libraries that made this possible.

Start using WHOMP to create balanced subgroups and explore the power of optimal matching and partitioning!

## Reference

If you use WHOMP in your research, please cite the following paper:

```latex
@article{xu2024whomp,
  title={WHOMP: Optimizing Randomized Controlled Trials via Wasserstein Homogeneity},
  author={Xu, Shizhou and Strohmer, Thomas},
  journal={arXiv preprint arXiv:2409.18504},
  year={2024}
}
```
