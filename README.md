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

![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/Constrained_Kmeans.png)

### Example: WHOMP Random

WHOMP also provides functionality to generate evenly dispersed subgroups with ''anti-clustering'' techniques.

```python
from WHOMP import WHOMP_Random

# Apply WHOMP Random Anti-Clustering
anti_index = WHOMP_Random(X_toy, 2)

# Plot WHOMP Random Anti-Clustering result
for i in range(3):
    plt.scatter(X_toy[np.where(anti_index == i)[0], 0], X_toy[np.where(anti_index == i)[0], 1])
plt.title('WHOMP Random Result')
plt.show()
```

![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/WHOMP_Random.png)


### Example: WHOMP Matching

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

![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/WHOMP_Matching.png)


## Test Cases

WHOMP has several built-in test cases that evaluate the performance of its partitioning methods in various scenarios:

### 1. Wasserstein Experiment on Gaussian Dataset

This test demonstrates how to calculate Wasserstein-2 distances between subsampled groups and the original dataset using Gaussian blobs.

```python
from WHOMP_Test import Wasserstein_Gaussian_experiment

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

![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/WHOMP_Gaussian_2.png)
![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/WHOMP_Gaussian_4.png)
![alt text](https://github.com/xushizhou/WHOMP/blob/main/images/WHOMP_Gaussian_6.png)


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
random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave, random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std, random_var_list, PS_var_list, anti_var_list, bary_var_list, random_mean_list, PS_mean_list, anti_mean_list, bary_mean_list = NPI_experiment(X, range(2, 7, 2), repetition = 500)

# Print average and standard deviations of errors
print("Average Errors:", random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave)
print("Standard Deviations:", random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std)

# Print variance of partition results
for i in range(3):
    print(f"Variances (bary_var_list[{i}]):", np.var(np.array(bary_var_list[i])),
            np.var(np.array(anti_var_list[i])), np.var(np.array(PS_var_list[i])), np.var(np.array(random_var_list[i])))

for i in range(3):
    print(f"Mean Variances (bary_mean_list[{i}]):", np.var(np.array(bary_mean_list[i])),
            np.var(np.array(anti_mean_list[i])), np.var(np.array(PS_mean_list[i])), np.var(np.array(random_mean_list[i])))
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
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from WHOMP_Test import run_WHOMP_MNIST_experiment

# data from sklearn datasets
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

# Extract data & target from the dataset
pixel_data, targets = data
targets = targets.astype(int)
x = pixel_data.values
y = targets.values

## Standardizing the data
standardized_data = StandardScaler().fit_transform(x)
print(standardized_data.shape)

# t-SNE is consumes a lot of memory so we shall use only a subset of our dataset. 
x_subset = x[0:120]
y_subset = y[0:120]
label = y_subset

tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=10, n_iter=5000).fit_transform(x_subset)

# Run the entropy experiment
ent_average_list, ent_std_list = run_WHOMP_MNIST_experiment(tsne, y_subset, repetition = 50)
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

We welcome contributions! Please contact me via my email if you would like to report issues or suggest features.

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
