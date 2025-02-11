import numpy as np
import random
import matplotlib.pyplot as plt
import ot
import networkx as nx
from sklearn.manifold import spectral_embedding
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
from WHOMP import WHOMP_Random, WHOMP_Matching, Pocock_Simon_minimization, RandomPartition, KMeansConstrained

def HoldoutSet(X, y, holdoutindex):
    X_holdout = X[holdoutindex == 1]
    X_comp = X[holdoutindex != 1]
    y_holdout = y[holdoutindex == 1]
    y_comp = y[holdoutindex != 1]
    return X_holdout, X_comp, y_holdout, y_comp

def RandomSubgroupSet(X, y, subgroupindex):
    shuffle = np.arange((len(set(subgroupindex))))
    random.shuffle(shuffle)
    X_holdout = X[subgroupindex == shuffle[1]]
    X_comp = X[subgroupindex == shuffle[0]]
    y_holdout = y[subgroupindex == shuffle[1]]
    y_comp = y[subgroupindex == shuffle[0]]
    return X_holdout, X_comp, y_holdout, y_comp

def RandomSubgroupSet_Complement(X, y, subgroupindex):
    shuffle = np.arange((len(set(subgroupindex))))
    random.shuffle(shuffle)
    X_holdout = X[subgroupindex == shuffle[1]]
    X_comp = X[subgroupindex != shuffle[1]]
    y_holdout = y[subgroupindex == shuffle[1]]
    y_comp = y[subgroupindex != shuffle[1]]
    return X_holdout, X_comp, y_holdout, y_comp

def logistic_regression_Gaussian_mixture_experiment(subgroup_number, repetition):
    random_error = []
    PS_error = []
    WHOMP_random_error = []
    WHOMP_matching_error = []
    
    for s in range(repetition):
        # data from mixture of Gaussians
        X_toy, truth_model = make_blobs(n_samples=60, centers=[(0,10), (-10,-5), (10,-5)], cluster_std=[4, 4, 4], random_state=42)
                        
        barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, 0.0000001)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        X_WHOMP_matching = X_toy[barysample_index[shuffle_index[0]], :]
        X_WHOMP_matching_comp = X_toy[barysample_index[shuffle_index[1]], :]
        y_WHOMP_matching = truth_model[barysample_index[shuffle_index[0]]]
        y_WHOMP_matching_comp = truth_model[barysample_index[shuffle_index[1]]]

        PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
        X_PS, X_PS_comp, y_PS, y_PS_comp = RandomSubgroupSet(X_toy, truth_model, PS_index)

        WHOMP_random_index = WHOMP_Random(X_toy, subgroup_number=subgroup_number)
        X_WHOMP_random, X_WHOMP_random_comp, y_WHOMP_random, y_WHOMP_random_comp = RandomSubgroupSet(X_toy, truth_model, WHOMP_random_index)

        random_index = RandomPartition(X_toy, subgroup_number=subgroup_number)
        X_sample, X_sample_comp, y_sample, y_sample_comp = RandomSubgroupSet(X_toy, truth_model, random_index)

        # Train logistic regression models
        clf_random = LogisticRegression(random_state=0).fit(X_sample, y_sample)
        clf_PS = LogisticRegression(random_state=0).fit(X_PS, y_PS)
        clf_WHOMP_random = LogisticRegression(random_state=0).fit(X_WHOMP_random, y_WHOMP_random)
        clf_WHOMP_matching = LogisticRegression(random_state=0).fit(X_WHOMP_matching, y_WHOMP_matching)

        # Evaluate models
        random_error.append(clf_random.score(X_sample_comp, y_sample_comp))
        PS_error.append(clf_PS.score(X_PS_comp, y_PS_comp))
        WHOMP_random_error.append(clf_WHOMP_random.score(X_WHOMP_random_comp, y_WHOMP_random_comp))
        WHOMP_matching_error.append(clf_WHOMP_matching.score(X_WHOMP_matching_comp, y_WHOMP_matching_comp))
    
    return random_error, PS_error, WHOMP_random_error, WHOMP_matching_error

def SVM_Gaussian_mixture_experiment(subgroup_number, repetition):
    random_error = []
    PS_error = []
    WHOMP_random_error = []
    WHOMP_matching_error = []
    
    for s in range(repetition):
        # data from mixture of Gaussians
        X_toy, truth_model = make_blobs(n_samples=60, centers=[(0, 10), (-10, -5), (10, -5)], cluster_std=[4, 2, 2], random_state=42)
    
        # WHOMP Matching Partition
        barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, 0.0000001)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        X_WHOMP_matching = X_toy[barysample_index[shuffle_index[0]], :]
        X_WHOMP_matching_comp = X_toy[barysample_index[shuffle_index[1]], :]
        y_WHOMP_matching = truth_model[barysample_index[shuffle_index[0]]]
        y_WHOMP_matching_comp = truth_model[barysample_index[shuffle_index[1]]]

        # Pocock-Simon Partition
        PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
        X_PS, X_PS_comp, y_PS, y_PS_comp = RandomSubgroupSet(X_toy, truth_model, PS_index)

        # WHOMP Random Partition
        WHOMP_random_index = WHOMP_Random(X_toy, subgroup_number=subgroup_number)
        X_WHOMP_random, X_WHOMP_random_comp, y_WHOMP_random, y_WHOMP_random_comp = RandomSubgroupSet(X_toy, truth_model, WHOMP_random_index)

        # Random Partition
        random_index = RandomPartition(X_toy, subgroup_number=subgroup_number)
        X_sample, X_sample_comp, y_sample, y_sample_comp = RandomSubgroupSet(X_toy, truth_model, random_index)

        # Train SVM models
        clf_random = svm.SVC().fit(X_sample, y_sample)
        clf_PS = svm.SVC().fit(X_PS, y_PS)
        clf_WHOMP_random = svm.SVC().fit(X_WHOMP_random, y_WHOMP_random)
        clf_WHOMP_matching = svm.SVC().fit(X_WHOMP_matching, y_WHOMP_matching)

        # Evaluate models
        random_error.append(clf_random.score(X_sample_comp, y_sample_comp))
        PS_error.append(clf_PS.score(X_PS_comp, y_PS_comp))
        WHOMP_random_error.append(clf_WHOMP_random.score(X_WHOMP_random_comp, y_WHOMP_random_comp))
        WHOMP_matching_error.append(clf_WHOMP_matching.score(X_WHOMP_matching_comp, y_WHOMP_matching_comp))
    
    return random_error, PS_error, WHOMP_random_error, WHOMP_matching_error


def LinearRegression_Gaussian_mixture_experiment(subgroup_number, repetition):
    random_error = []
    PS_error = []
    WHOMP_random_error = []
    WHOMP_matching_error = []
    
    for s in range(repetition):
        # Data from mixture of Gaussians
        X_toy, truth_model = make_blobs(n_samples=60, centers=[(0, 10), (-10, -5), (10, -5)], cluster_std=[4, 2, 2], random_state=42)
    
        # WHOMP Matching Partition
        barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, 0.0000001)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        X_WHOMP_matching = X_toy[barysample_index[shuffle_index[0]], :]
        X_WHOMP_matching_comp = X_toy[barysample_index[shuffle_index[1]], :]
        y_WHOMP_matching = truth_model[barysample_index[shuffle_index[0]]]
        y_WHOMP_matching_comp = truth_model[barysample_index[shuffle_index[1]]]

        # Pocock-Simon Partition
        PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
        X_PS, X_PS_comp, y_PS, y_PS_comp = RandomSubgroupSet(X_toy, truth_model, PS_index)

        # WHOMP Random Partition
        WHOMP_random_index = WHOMP_Random(X_toy, subgroup_number=subgroup_number)
        X_WHOMP_random, X_WHOMP_random_comp, y_WHOMP_random, y_WHOMP_random_comp = RandomSubgroupSet(X_toy, truth_model, WHOMP_random_index)

        # Random Partition
        random_index = RandomPartition(X_toy, subgroup_number=subgroup_number)
        X_sample, X_sample_comp, y_sample, y_sample_comp = RandomSubgroupSet(X_toy, truth_model, random_index)

        # Train Linear Regression models
        clf_random = LinearRegression().fit(X_sample[:, 0].reshape(-1, 1), X_sample[:, 1])
        clf_PS = LinearRegression().fit(X_PS[:, 0].reshape(-1, 1), X_PS[:, 1])
        clf_WHOMP_random = LinearRegression().fit(X_WHOMP_random[:, 0].reshape(-1, 1), X_WHOMP_random[:, 1])
        clf_WHOMP_matching = LinearRegression().fit(X_WHOMP_matching[:, 0].reshape(-1, 1), X_WHOMP_matching[:, 1])

        # Predictions
        pred_random = clf_random.predict(X_sample_comp[:, 0].reshape(-1, 1))
        pred_PS = clf_PS.predict(X_PS_comp[:, 0].reshape(-1, 1))
        pred_WHOMP_random = clf_WHOMP_random.predict(X_WHOMP_random_comp[:, 0].reshape(-1, 1))
        pred_WHOMP_matching = clf_WHOMP_matching.predict(X_WHOMP_matching_comp[:, 0].reshape(-1, 1))

        # Errors (using L2 norm divided by number of samples)
        random_error.append(np.linalg.norm(pred_random - X_sample_comp[:, 1]) / len(pred_random))
        PS_error.append(np.linalg.norm(pred_PS - X_PS_comp[:, 1]) / len(pred_PS))
        WHOMP_random_error.append(np.linalg.norm(pred_WHOMP_random - X_WHOMP_random_comp[:, 1]) / len(pred_WHOMP_random))
        WHOMP_matching_error.append(np.linalg.norm(pred_WHOMP_matching - X_WHOMP_matching_comp[:, 1]) / len(pred_WHOMP_matching))
    
    return random_error, PS_error, WHOMP_random_error, WHOMP_matching_error


import numpy as np
import matplotlib.pyplot as plt
import ot
from sklearn.datasets import make_blobs
from WHOMP import WHOMP_Matching, Pocock_Simon_minimization, WHOMP_Random, RandomPartition


# Experiment setup: comparing WHOMP, Random, Pocock-Simon, and Anti-Clustering partitioning methods
def Wasserstein_Gaussian_experiment():
    random_error_ave = []
    PS_error_ave = []
    anticluster_error_ave = []
    WHOMP_error_ave = []

    random_error_std = []
    PS_error_std = []
    anticluster_error_std = []
    WHOMP_error_std = []

    random_error_list = []
    PS_error_list = []
    anticluster_error_list = []
    WHOMP_error_list = []

    random_mean_list = []
    PS_mean_list = []
    anti_mean_list = []
    bary_mean_list = []

    random_var_list = []
    PS_var_list = []
    anti_var_list = []
    bary_var_list = []

    for k in range(2, 7, 2):  # Running for 2, 4, and 6 subgroups
        random_error = []
        PS_error = []
        anticluster_error = []
        WHOMP_error = []

        mean_random = []
        var_random = []

        mean_PS = []
        var_PS = []

        mean_anti = []
        var_anti = []

        mean_bary = []
        var_bary = []

        for s in range(100):  # 100 repetitions per subgroup number
            # Generate data from a mixture of Gaussians
            X_toy, truth_model = make_blobs(n_samples=60, centers=[(0, 10), (-10, -5), (10, -5)], cluster_std=[3, 3, 3])

            subgroup_number = k

            # Apply different partitioning methods
            barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, 0.00000001)
            PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
            anticluster_index = WHOMP_Random(X_toy, subgroup_number=subgroup_number)
            random_index = RandomPartition(X_toy, subgroup_number=subgroup_number)

            # Compute Wasserstein distances (W2) for each method
            X_bary = X_toy[barysample_index[0], :]
            density_sensitive = np.ones(X_bary.shape[0]) / X_bary.shape[0]
            density_X = np.ones(X_toy.shape[0]) / X_toy.shape[0]

            # Random partition error
            eps_random = 0
            for i in range(subgroup_number):
                X_random = X_toy[random_index == i, :]
                cost_matrix_random = ot.dist(X_toy, X_random)
                plan_random = np.array(ot.emd(density_X, density_sensitive, cost_matrix_random))
                eps_random += np.sum(cost_matrix_random * plan_random)
                mean_random.append(np.average(X_random))
                var_random.append(np.var(X_random))

            # Pocock-Simon error
            eps_PS = 0
            for i in range(subgroup_number):
                X_ps = X_toy[PS_index == i, :]
                cost_matrix_PS = ot.dist(X_toy, X_ps)
                plan_PS = np.array(ot.emd(density_X, density_sensitive, cost_matrix_PS))
                eps_PS += np.sum(cost_matrix_PS * plan_PS)
                mean_PS.append(np.average(X_ps))
                var_PS.append(np.var(X_ps))

            # WHOMP Random error
            eps_anti = 0
            for i in range(subgroup_number):
                X_anti = X_toy[anticluster_index == i, :]
                cost_matrix_anti = ot.dist(X_toy, X_anti)
                plan_anti = np.array(ot.emd(density_X, density_sensitive, cost_matrix_anti))
                eps_anti += np.sum(cost_matrix_anti * plan_anti)
                mean_anti.append(np.average(X_anti))
                var_anti.append(np.var(X_anti))

            # WHOMP Barycenter Matching error
            eps_bary = 0
            for i in range(subgroup_number):
                X_bary = X_toy[barysample_index[i], :]
                cost_matrix_bary = ot.dist(X_toy, X_bary)
                plan_bary = np.array(ot.emd(density_X, density_sensitive, cost_matrix_bary))
                eps_bary += np.sum(cost_matrix_bary * plan_bary)
                mean_bary.append(np.average(X_bary))
                var_bary.append(np.var(X_bary))

            # Record errors for this repetition
            random_error.append(np.sqrt(eps_random / subgroup_number))
            PS_error.append(np.sqrt(eps_PS / subgroup_number))
            anticluster_error.append(np.sqrt(eps_anti / subgroup_number))
            WHOMP_error.append(np.sqrt(eps_bary / subgroup_number))

        # Store results for this subgroup number
        random_error_list.append(random_error)
        PS_error_list.append(PS_error)
        anticluster_error_list.append(anticluster_error)
        WHOMP_error_list.append(WHOMP_error)

        random_error_ave.append(np.average(random_error))
        PS_error_ave.append(np.average(PS_error))
        anticluster_error_ave.append(np.average(anticluster_error))
        WHOMP_error_ave.append(np.average(WHOMP_error))

        random_error_std.append(np.std(random_error))
        PS_error_std.append(np.std(PS_error))
        anticluster_error_std.append(np.std(anticluster_error))
        WHOMP_error_std.append(np.std(WHOMP_error))

        random_mean_list.append(mean_random)
        PS_mean_list.append(mean_PS)
        anti_mean_list.append(mean_anti)
        bary_mean_list.append(mean_bary)

        random_var_list.append(var_random)
        PS_var_list.append(var_PS)
        anti_var_list.append(var_anti)
        bary_var_list.append(var_bary)

        # Plot histogram of errors for this subgroup number
        fig = plt.figure(figsize=(10, 7))
        xbins = np.arange(0, 10, 0.2)

        plt.hist(random_error, bins=xbins, density=False, weights=1 / len(random_error) * np.ones(len(random_error)), histtype='step', color='#377eb8', label='Random partition')
        plt.hist(PS_error, bins=xbins, density=False, weights=1 / len(PS_error) * np.ones(len(PS_error)), histtype='step', color='#4daf4a', label='Pocock & Simon')
        plt.hist(anticluster_error, bins=xbins, density=False, weights=1 / len(anticluster_error) * np.ones(len(anticluster_error)), histtype='step', color='#a65628', label='WHOMP: random')
        plt.hist(WHOMP_error, bins=xbins, density=False, weights=1 / len(WHOMP_error) * np.ones(len(WHOMP_error)), histtype='step', color='#999999', label='WHOMP: barycenter matching')

        plt.xlabel('W2 distance between subgroup and original sample', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.legend(fontsize=15, title='Partition methods')

        plt.title(f"{k} Subgroups")
        plt.show()

    return random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave, random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std

def plot_kmeans_constrained(X_toy, n_clusters=3):
    """
    Apply balanced K-means clustering (KMeansConstrained) and visualize the results.
    """
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=20, size_max=20, random_state=0)
    clf.fit_predict(X_toy)
    
    # Plot the KMeansConstrained clustering result
    plt.scatter(X_toy[:, 0], X_toy[:, 1], c=clf.labels_, cmap='viridis', marker='o')
    plt.title('Balanced K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_whomp_random(X_toy, subgroup_number=2):
    """
    Apply WHOMP Random partitioning and visualize the results.
    """
    anti_index = WHOMP_Random(X_toy, subgroup_number)
    for i in range(subgroup_number):
        plt.scatter(X_toy[np.where(anti_index == i)[0], 0], X_toy[np.where(anti_index == i)[0], 1], label = f" subgroup {i + 1}")
    
    plt.title(f"WHOMP Random")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend(loc="upper right")
    plt.savefig('whomp_random_distributions.png')
    plt.show()

def plot_whomp_matching(X_toy, subgroup_number=2, threshold=0.0000001):
    """
    Apply WHOMP Matching partitioning and visualize the results.
    """
    barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, threshold)
    
    for i in range(subgroup_number):
        X = np.concatenate([X_toy[barysample_index[i], :], np.expand_dims(barycenter[i, :], axis=0)], axis=0)
        plt.scatter(X_toy[barysample_index[i], 0], X_toy[barysample_index[i], 1], label = f" subgroup {i + 1}")
    
    plt.title(f"WHOMP Matching")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend(loc="upper right")
    plt.savefig('whomp_matching_distributions.png')
    plt.show()

def plot_pocock_simon(X_toy, subgroup_number=2):
    """
    Apply Pocock & Simon minimization partitioning and visualize the results.
    """
    PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
    
    for i in range(subgroup_number): 
        plt.scatter(X_toy[np.where(PS_index == i)[0], 0], X_toy[np.where(PS_index == i)[0], 1], label = f" subgroup {i + 1}")
    
    plt.title(f"Pocock & Simon's covariate-adaptive randomization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend(loc="upper right")
    plt.savefig('pocock_simon_distributions.png')
    plt.show()

def plot_random(X_toy, subgroup_number=2):
    """
    Apply random partition and visualize the results.
    """
    PS_index = RandomPartition(X_toy, subgroup_number)
    
    for i in range(subgroup_number): 
        plt.scatter(X_toy[np.where(PS_index == i)[0], 0], X_toy[np.where(PS_index == i)[0], 1], label = f" subgroup {i + 1}")
    
    plt.title(f"Random partition")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend(loc="upper right")
    plt.savefig('random_partition_distributions.png')
    plt.show()

def SBM_spectrum_experiment(subgroup_number, repetition):
    """
    Run SBM spectrum experiment using different partitioning methods (WHOMP, Pocock-Simon, Random).
    """
    random_error_ave = []
    PS_error_ave = []
    WHOMP_random_error_ave = []
    WHOMP_matching_error_ave = []

    random_error_std = []
    PS_error_std = []
    WHOMP_random_error_std = []
    WHOMP_matching_error_std = []
    
    for s in range(repetition):
        sizes = [20, 20, 20]
        probs = [[0.8, 0.4, 0.2], [0.4, 0.8, 0.4], [0.2, 0.4, 0.8]]
        G = nx.stochastic_block_model(sizes, probs, seed=0)

        spec_0 = nx.laplacian_spectrum(G)
        density_0 = np.ones(len(spec_0)) / len(spec_0)
        adj_matrix = nx.to_numpy_array(G)
        G_embedded = spectral_embedding(adj_matrix, n_components=5, random_state=42, norm_laplacian=True)
        shuffle_index = np.arange(subgroup_number)

        # WHOMP Matching Partition
        barycenter, barysample_index = WHOMP_Matching(G_embedded, subgroup_number, 0.0000001)
        np.random.shuffle(shuffle_index)
        error = []
        for i in range(subgroup_number):
            subgraph = G.subgraph(barysample_index[shuffle_index[i]])
            spec_1 = nx.laplacian_spectrum(subgraph)
            density_1 = np.ones(len(spec_1)) / len(spec_1)
            cost_matrix = ot.dist(spec_0.reshape((len(spec_0), 1)), spec_1.reshape((len(spec_1), 1)))
            plan = np.array(ot.emd(density_0, density_1, cost_matrix))
            error.append(np.sqrt(np.sum(plan * cost_matrix)))
        WHOMP_matching_error_ave.append(np.average(error))
        WHOMP_matching_error_std.append(np.std(error))

        # WHOMP Random Partition
        anti_index = WHOMP_Random(G_embedded, subgroup_number=subgroup_number)
        np.random.shuffle(shuffle_index)
        error = []
        for i in range(subgroup_number):
            subgraph = G.subgraph(np.where(anti_index == shuffle_index[i])[0])
            spec_1 = nx.laplacian_spectrum(subgraph)
            density_1 = np.ones(len(spec_1)) / len(spec_1)
            cost_matrix = ot.dist(spec_0.reshape((len(spec_0), 1)), spec_1.reshape((len(spec_1), 1)))
            plan = np.array(ot.emd(density_0, density_1, cost_matrix))
            error.append(np.sqrt(np.sum(plan * cost_matrix)))
        WHOMP_random_error_ave.append(np.average(error))
        WHOMP_random_error_std.append(np.std(error))

        # Pocock & Simon Partition
        PS_index = Pocock_Simon_minimization(G_embedded, subgroup_number)
        np.random.shuffle(shuffle_index)
        error = []
        for i in range(subgroup_number):
            subgraph = G.subgraph(np.where(PS_index == shuffle_index[i])[0])
            spec_1 = nx.laplacian_spectrum(subgraph)
            density_1 = np.ones(len(spec_1)) / len(spec_1)
            cost_matrix = ot.dist(spec_0.reshape((len(spec_0), 1)), spec_1.reshape((len(spec_1), 1)))
            plan = np.array(ot.emd(density_0, density_1, cost_matrix))
            error.append(np.sqrt(np.sum(plan * cost_matrix)))
        PS_error_ave.append(np.average(error))
        PS_error_std.append(np.std(error))

        # Random Partition
        random_index = RandomPartition(G_embedded, subgroup_number=subgroup_number)
        np.random.shuffle(shuffle_index)
        error = []
        for i in range(subgroup_number):
            subgraph = G.subgraph(np.where(random_index == shuffle_index[i])[0])
            spec_1 = nx.laplacian_spectrum(subgraph)
            density_1 = np.ones(len(spec_1)) / len(spec_1)
            cost_matrix = ot.dist(spec_0.reshape((len(spec_0), 1)), spec_1.reshape((len(spec_1), 1)))
            plan = np.array(ot.emd(density_0, density_1, cost_matrix))
            error.append(np.sqrt(np.sum(plan * cost_matrix)))
        random_error_ave.append(np.average(error))
        random_error_std.append(np.std(error))

    # Average and standard deviation results
    random_ave = [np.average(random_error_ave), np.std(random_error_ave)]
    PS_ave = [np.average(PS_error_ave), np.std(PS_error_ave)]
    WHOMP_random_ave = [np.average(WHOMP_random_error_ave), np.std(WHOMP_random_error_ave)]
    WHOMP_matching_ave = [np.average(WHOMP_matching_error_ave), np.std(WHOMP_matching_error_ave)]

    random_std = [np.average(random_error_std), np.std(random_error_std)]
    PS_std = [np.average(PS_error_std), np.std(PS_error_std)]
    WHOMP_random_std = [np.average(WHOMP_random_error_std), np.std(WHOMP_random_error_std)]
    WHOMP_matching_std = [np.average(WHOMP_matching_error_std), np.std(WHOMP_matching_error_std)]

    Ave_list = [random_ave, PS_ave, WHOMP_random_ave, WHOMP_matching_ave]
    Std_list = [random_std, PS_std, WHOMP_random_std, WHOMP_matching_std]
    
    return Ave_list, Std_list

def NPI_experiment(X, subgroup_number_list, repetition):
    random_error_ave = []
    PS_error_ave = []
    anticluster_error_ave = []
    WHOMP_error_ave = []

    random_error_std = []
    PS_error_std = []
    anticluster_error_std = []
    WHOMP_error_std = []

    random_error_list = []
    PS_error_list = []
    anticluster_error_list = []
    WHOMP_error_list = []

    random_mean_list = []
    PS_mean_list = []
    anti_mean_list = []
    bary_mean_list = []

    random_var_list = []
    PS_var_list = []
    anti_var_list = []
    bary_var_list = []

    for k in subgroup_number_list:
        random_error = []
        PS_error = []
        anticluster_error = []
        WHOMP_error = []

        mean_random = []
        var_random = []
        mean_PS = []
        var_PS = []
        mean_anti = []
        var_anti = []
        mean_bary = []
        var_bary = []

        for s in range(repetition):
            # Shuffle and subsample the data
            index_shuffle = np.arange(X.shape[0])
            np.random.shuffle(index_shuffle)
            X_toy = X[index_shuffle[:60], :]  # Taking a subsample of size 60
            truth_model = np.zeros(X_toy.shape[0])  # Placeholder truth model (can adjust as needed)
            subgroup_number = k

            # Apply different partitioning methods
            barycenter, barysample_index = WHOMP_Matching(X_toy, subgroup_number, 0.00000001)
            PS_index = Pocock_Simon_minimization(X_toy, subgroup_number)
            anticluster_index = WHOMP_Random(X_toy, subgroup_number=subgroup_number)
            random_index = RandomPartition(X_toy, subgroup_number=subgroup_number)

            # Define density distributions for comparison
            X_bary = X_toy[barysample_index[0], :]
            density_sensitive = np.ones(X_bary.shape[0]) / X_bary.shape[0]
            density_X = np.ones(X_toy.shape[0]) / X_toy.shape[0]

            # Calculate Wasserstein-2 distances and statistics for each method
            eps_random, eps_PS, eps_anti, eps_bary = 0, 0, 0, 0

            for i in range(subgroup_number):
                # Random Partition
                X_random = X_toy[random_index == i, :]
                cost_matrix_random = ot.dist(X_toy, X_random)
                plan_random = np.array(ot.emd(density_X, density_sensitive, cost_matrix_random))
                eps_random += np.sum(cost_matrix_random * plan_random)
                mean_random.append(np.average(X_random))
                var_random.append(np.var(X_random))

                # Pocock & Simon Partition
                X_ps = X_toy[PS_index == i, :]
                cost_matrix_PS = ot.dist(X_toy, X_ps)
                plan_PS = np.array(ot.emd(density_X, density_sensitive, cost_matrix_PS))
                eps_PS += np.sum(cost_matrix_PS * plan_PS)
                mean_PS.append(np.average(X_ps))
                var_PS.append(np.var(X_ps))

                # WHOMP Random Partition
                X_anti = X_toy[anticluster_index == i, :]
                cost_matrix_anti = ot.dist(X_toy, X_anti)
                plan_anti = np.array(ot.emd(density_X, density_sensitive, cost_matrix_anti))
                eps_anti += np.sum(cost_matrix_anti * plan_anti)
                mean_anti.append(np.average(X_anti))
                var_anti.append(np.var(X_anti))

                # WHOMP Barycenter Partition
                X_bary = X_toy[barysample_index[i], :]
                cost_matrix_bary = ot.dist(X_toy, X_bary)
                plan_bary = np.array(ot.emd(density_X, density_sensitive, cost_matrix_bary))
                eps_bary += np.sum(cost_matrix_bary * plan_bary)
                mean_bary.append(np.average(X_bary))
                var_bary.append(np.var(X_bary))

            random_error.append(np.sqrt(eps_random / subgroup_number))
            PS_error.append(np.sqrt(eps_PS / subgroup_number))
            anticluster_error.append(np.sqrt(eps_anti / subgroup_number))
            WHOMP_error.append(np.sqrt(eps_bary / subgroup_number))

        # Store the results for each subgroup number
        random_error_list.append(random_error)
        PS_error_list.append(PS_error)
        anticluster_error_list.append(anticluster_error)
        WHOMP_error_list.append(WHOMP_error)

        random_error_ave.append(np.average(random_error))
        PS_error_ave.append(np.average(PS_error))
        anticluster_error_ave.append(np.average(anticluster_error))
        WHOMP_error_ave.append(np.average(WHOMP_error))

        random_error_std.append(np.std(random_error))
        PS_error_std.append(np.std(PS_error))
        anticluster_error_std.append(np.std(anticluster_error))
        WHOMP_error_std.append(np.std(WHOMP_error))

        random_mean_list.append(mean_random)
        PS_mean_list.append(mean_PS)
        anti_mean_list.append(mean_anti)
        bary_mean_list.append(mean_bary)

        random_var_list.append(var_random)
        PS_var_list.append(var_PS)
        anti_var_list.append(var_anti)
        bary_var_list.append(var_bary)

        # Plot the histogram of errors
        fig = plt.figure(figsize=(10, 7))
        xbins = np.arange(1.8 + (k - 2) * 0.25, 2.6 + (k - 2) * 0.25, 0.02)

        plt.hist(random_error, bins=xbins, density=False, weights=1 / len(random_error) * np.ones(len(random_error)),
                 histtype='step', color='#377eb8', label='Random partition')
        plt.hist(PS_error, bins=xbins, density=False, weights=1 / len(PS_error) * np.ones(len(PS_error)),
                 histtype='step', color='#4daf4a', label='Covariate-adaptive randomization: Pocock & Simon')
        plt.hist(anticluster_error, bins=xbins, density=False, weights=1 / len(anticluster_error) * np.ones(len(anticluster_error)),
                 histtype='step', color='#a65628', label='WHOMP: random')
        plt.hist(WHOMP_error, bins=xbins, density=False, weights=1 / len(WHOMP_error) * np.ones(len(WHOMP_error)),
                 histtype='step', color='#999999', label='WHOMP: barycenter matching')

        plt.xlabel('Wasserstein-2 distance between resulted subgroup and the original sample')
        plt.ylabel('frequency')
        plt.legend(fontsize='medium', title='Subsampling/partition methods:')
        plt.title(f"{k} Subgroups")

        # Show the plot
        plt.show()

    return random_error_ave, PS_error_ave, anticluster_error_ave, WHOMP_error_ave, random_error_std, PS_error_std, anticluster_error_std, WHOMP_error_std, random_var_list, PS_var_list, anti_var_list, bary_var_list, random_mean_list, PS_mean_list, anti_mean_list, bary_mean_list
           

def normalized_entropy(data):
    """Calculates entropy of the passed numpy array
    """
    unique, counts = np.unique(data, return_counts=True)
    prob_vec = counts/len(data)
    base = len(unique)
    normalized_entropy = -np.sum(prob_vec * np.log(prob_vec)) / np.log(base)
    return normalized_entropy

def normalized_entropy(labels):
    """Calculate normalized entropy of a set of labels."""
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    max_entropy = np.log2(len(np.unique(labels)))
    return entropy / max_entropy if max_entropy > 0 else 0

def WHOMP_MNIST_experiment(tsne, y_subset, subgroup_number, repetition):
    random_ent = []
    PS_ent = []
    WHOMP_Random_ent = []
    WHOMP_Matching_ent = []
    
    for s in range(repetition):
        barycenter, barysample_index = WHOMP_Matching(tsne, subgroup_number , 0.0000001)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        for i in range(subgroup_number):
            ent = normalized_entropy(y_subset[barysample_index[shuffle_index[i]]])
            WHOMP_Matching_ent.append(ent)

        anticluster_index = WHOMP_Random(tsne, subgroup_number=subgroup_number)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        for i in range(subgroup_number):
            ent = normalized_entropy(y_subset[anticluster_index == shuffle_index[i]])
            WHOMP_Random_ent.append(ent)

        PS_index = Pocock_Simon_minimization(tsne, subgroup_number)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        for i in range(subgroup_number):
            ent = normalized_entropy(y_subset[PS_index == shuffle_index[i]])
            PS_ent.append(ent)

        random_index = RandomPartition(tsne, subgroup_number=subgroup_number)
        shuffle_index = np.arange(subgroup_number)
        np.random.shuffle(shuffle_index)
        for i in range(subgroup_number):
            ent = normalized_entropy(y_subset[random_index == shuffle_index[i]])
            random_ent.append(ent)
    
    return random_ent, PS_ent, WHOMP_Random_ent, WHOMP_Matching_ent

def run_WHOMP_MNIST_experiment(tsne, y_subset, repetition):
    """
    Run the entropy experiment for MNIST data.
    """
    data_entropy = normalized_entropy(y_subset)

    ent_average_list = []
    ent_std_list = []

    for subgroup_number in range(2,7,2):
        random_ent, PS_ent, WHOMP_Random_ent, WHOMP_Matching_ent = WHOMP_MNIST_experiment(tsne, y_subset, subgroup_number, repetition = repetition)
        ent_average_list.append((np.average(random_ent), np.average(PS_ent), np.average(WHOMP_Random_ent), np.average(WHOMP_Matching_ent)))
        ent_std_list.append((np.std(random_ent), np.std(PS_ent), np.std(WHOMP_Random_ent), np.std(WHOMP_Matching_ent)))

    # Output the results
    print("Data entropy:", data_entropy)
    print("\nAverage Entropy List:", ent_average_list)
    print("\nStandard Deviation List:", ent_std_list)

    return ent_average_list, ent_std_list
