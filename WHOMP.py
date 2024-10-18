# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import scipy.stats as st
import scipy
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import check_random_state, check_array, as_float_array, check_is_fitted
from joblib import Parallel, delayed
import random
import warnings
import ot


def k_means_constrained(X, n_clusters, size_min=None, size_max=None, init='k-means++',
                        n_init=10, max_iter=300, verbose=False,
                        tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
                        return_n_iter=False):

    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X, copy=copy_x)
    #tol = _tolerance(X, tol)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        #_validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_constrained_single(
                X, n_clusters,
                size_min=size_min, size_max=size_max,
                max_iter=max_iter, init=init, verbose=verbose, tol=tol,
                x_squared_norms=x_squared_norms, random_state=random_state)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_constrained_single)(X, n_clusters,
                                               size_min=size_min, size_max=size_max,
                                               max_iter=max_iter, init=init,
                                               verbose=verbose, tol=tol,
                                               x_squared_norms=x_squared_norms,
                                               # Change seed to ensure variety
                                               random_state=seed)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


def kmeans_constrained_single(X, n_clusters, size_min=None, size_max=None,
                              max_iter=300, init='k-means++',
                              verbose=False, x_squared_norms=None,
                              random_state=None, tol=1e-4):

    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its closer center for reallocation in case of ties
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

    # Determine min and max sizes if non given
    if size_min is None:
        size_min = 0
    if size_max is None:
        size_max = n_samples  # Number of data points

    # Check size min and max
    if not ((size_min >= 0) and (size_min <= n_samples)
            and (size_max >= 0) and (size_max <= n_samples)):
        raise ValueError("size_min and size_max must be a positive number smaller "
                         "than the number of data points or `None`")
    if size_max < size_min:
        raise ValueError("size_max must be larger than size_min")
    if size_min * n_clusters > n_samples:
        raise ValueError("The product of size_min and n_clusters cannot exceed the number of samples (X)")
    if size_max * n_clusters < n_samples:
        raise ValueError("The product of size_max and n_clusters must be larger than or equal the number of samples (X)")

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # E-step of EM
        labels, inertia = \
            _labels_constrained(X, centers, size_min, size_max)

        # M-step of EM
        if sp.issparse(X):
            centers = _centers_sparse(X, labels, n_clusters, distances)
        else:
            centers = _centers_dense(X, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels match cluster centers
        best_labels, best_inertia = \
            _labels_constrained(X, centers, size_min, size_max, distances=distances)

    return best_labels, best_inertia, best_centers, i + 1


def _centers_dense(X, labels, n_clusters, distances):
    # Compute the mean of each cluster
    return np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

def _centers_sparse(X, labels, n_clusters, distances):
    # Compute the mean of each cluster for sparse matrices
    centers = []
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if sp.issparse(cluster_points):
            cluster_mean = np.asarray(cluster_points.mean(axis=0)).ravel()
        else:
            cluster_mean = cluster_points.mean(axis=0)
        centers.append(cluster_mean)
    return np.array(centers)


def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None):
    """
    Compute the initial centroids
    """

    string_types = str
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, string_types) and init == 'k-means++':
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif isinstance(init, string_types) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        centers = np.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(X, k, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    #_validate_center_shape(X, k, centers)
    return centers

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """
    Init n_clusters seeds according to k-means++
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):

    # sum is as unstable as cumsum for numpy < 1.9
    np_version = _parse_version(np.__version__)
    if np_version < (1, 9):
        return np.cumsum(arr, axis=axis, dtype=np.float64)

    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out

def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):

    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)

def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):

    X, Y, dtype_float = _return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype, estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y

def safe_sparse_dot(a, b, dense_output=False):
    
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)
    

def _return_float_dtype(X, Y):
 
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float

    return X, Y, dtype


def _labels_constrained(X, centers, size_min, size_max):

    C = centers

    # Distances to each centre C. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    D = ot.dist(X, C)

    labels = solve_optimal_transport(X, C, D, size_min, size_max)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # Change distances in-place
    inertia = D[np.arange(D.shape[0]), labels].sum()

    return labels, inertia


def solve_optimal_transport(X, C, D, size_min, size_max):

    n_X = X.shape[0]
    n_C = C.shape[0]

    # Define supply and demand arrays
    supply = np.ones(n_X)  # Each supply node has a supply of 1
    demand = np.ones(n_C) * size_min  # Each demand node has a minimum demand of size_min

    # The supply should match the total demand, so adjust demand
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)

    if total_supply < total_demand:
        raise ValueError("Total supply is less than total demand. Adjust the supply or demand.")

    # Compute the cost matrix
    cost_matrix = D

    # Solve the optimal transport problem
    transport_plan = ot.emd(supply, demand, cost_matrix)

    # Find the exact assignment
    assignments = np.argmax(transport_plan, axis=1)

    return assignments


class KMeansConstrained(KMeans):

    def __init__(self, n_clusters=8, size_min=None, size_max=None, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 verbose=False, random_state=None, copy_x=True, n_jobs=1):

        self.size_min = size_min
        self.size_max = size_max

        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         verbose=verbose, random_state=random_state, copy_x=copy_x)

    def fit(self, X, y=None):
        
        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        random_state = check_random_state(self.random_state)
        'X = self._check_fit_data(X)'

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_constrained(
                X, n_clusters=self.n_clusters,
                size_min=self.size_min, size_max=self.size_max,
                init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                return_n_iter=True)
        return self

    def predict(self, X, size_min='init', size_max='init'):

        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        if size_min == 'init':
            size_min = self.size_min
        if size_max == 'init':
            size_max = self.size_max

        n_clusters = self.n_clusters
        n_samples = X.shape[0]

        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)

        # Allocate memory to store the distances for each sample to its closer center for reallocation in case of ties
        distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

        # Determine min and max sizes if non given
        if size_min is None:
            size_min = 0
        if size_max is None:
            size_max = n_samples  # Number of data points

        # Check size min and max
        if not ((size_min >= 0) and (size_min <= n_samples)
                and (size_max >= 0) and (size_max <= n_samples)):
            raise ValueError("size_min and size_max must be a positive number smaller "
                             "than the number of data points or `None`")
        if size_max < size_min:
            raise ValueError("size_max must be larger than size_min")
        if size_min * n_clusters > n_samples:
            raise ValueError("The product of size_min and n_clusters cannot exceed the number of samples (X)")

        labels, inertia = \
            _labels_constrained(X, self.cluster_centers_, size_min, size_max, distances=distances)

        return labels

    def fit_predict(self, X, y=None):

        return self.fit(X).labels_


# Additional helper functions/classes
def RandomPartition(X_toy, subgroup_number):

    random_index = np.zeros(len(X_toy))
    index = np.arange(len(X_toy))
    random.shuffle(index)
    random_subgroup = np.array_split(index, subgroup_number)
    for i in range(subgroup_number):
        random_index[random_subgroup[i]] = i

    return random_index

def Barycenter_Fixed_Point_LP(data, sensitive, threshhold):

    ## Initialization
    data_card, feature_dim = data.shape
    # Pick centroid as the overall average
    data_mean = np.average(data, axis = 0)
    sensitive_var_list = []
    sensitive_group_card_list = []
    # Pick variance as the smallest L2 norm among all sensitive groups
    sensitive_label = list(set(sensitive))
    sensitive_card = len(sensitive_label)
    for i in range(sensitive_card):
        sensitive_group = data[sensitive == sensitive_label[i],:]
        sensitive_group_card_list.append(sensitive_group.shape[0])
        sensitive_var_list.append(np.average(np.linalg.norm(sensitive_group,axis = 1)**2)**0.5)
    if len(set(sensitive_group_card_list)) > 1:
        return "Sensitive groups require to share the same cardinality."
        
    sensitive_group_card = sensitive_group_card_list[0]
    min_var = min(sensitive_var_list)
    # sample from Gaussian as initilization
    X_bar = np.random.multivariate_normal(data_mean, min_var*np.identity(feature_dim), sensitive_group_card)

    ## Iterative method to find the barycenter
    eps = threshhold + 1
    max_iter = 5000
    iter = 0
    while eps > threshhold and iter < max_iter:
        X_bar_new = np.zeros(X_bar.shape)
        X_bar_index = np.arange(sensitive_group_card)
        # Find the OT matching from X_bar to each sensitive group, then find the centroid of the matched points to update the barycenter
        match_list = []
        for i in range(sensitive_card):
            sensitive_group_index = np.where(sensitive == sensitive_label[i])[0]
            sensitive_group = data[sensitive_group_index,:]
            density_group = density_Xbar = np.ones(sensitive_group_card)
            cost_matrix = ot.dist(X_bar, sensitive_group)
            plan = np.array(ot.emd(density_Xbar, density_group, cost_matrix),dtype=int)
            match_index = plan @ X_bar_index
            match_list.append(sensitive_group_index[match_index])
            for j in range(sensitive_group_card):
                X_bar_new[j,:] += sensitive_group[match_index[j],:]
        X_bar_new = X_bar_new/sensitive_card

        # Update the distance between X_bar and the new one for the stop criteria
        cost_matrix_eps = ot.dist(X_bar, X_bar_new)
        plan_eps = np.array(ot.emd(density_Xbar/sensitive_group_card, density_group/sensitive_group_card, cost_matrix))
        eps = np.sqrt(np.sum(cost_matrix_eps*plan_eps))
        iter += 1
        X_bar = X_bar_new
        match_list = (np.array(match_list)).T
    
    return X_bar, match_list

def WHOMP_Matching(data, subgroup_number, threshold):

    sample_size = data.shape[0]
    clf = KMeansConstrained(n_clusters= int(sample_size/subgroup_number),size_min=subgroup_number,size_max=subgroup_number,n_init = 500, max_iter = 10000, tol = 0.000000001, random_state=None)
    clf.fit_predict(data)
    label = clf.labels_
    barycenter, OT_plan_list = Barycenter_Fixed_Point_LP(data, label, threshold)
    matched_list = OT_plan_list

    return barycenter, matched_list

def WHOMP_Random(data, subgroup_number):

    sample_size = data.shape[0]
    clf = KMeansConstrained(n_clusters= int(sample_size/subgroup_number),size_min=subgroup_number,size_max=subgroup_number,n_init = 500, max_iter = 10000, tol = 0.000000001, random_state=None)
    clf.fit_predict(data)
    label = clf.labels_
    anticluster_index = np.zeros(sample_size)-1
    for i in range(len(set(label))):
        index_i = np.where(label == i)[0]
        random.shuffle(index_i)
        index_i_select = np.array_split(index_i, subgroup_number)
        for j in range(subgroup_number):
            anticluster_index[index_i_select[j]] = j

    return anticluster_index

class MinimizationAlgorithm:
    def __init__(self, num_treatments, max_group_size):
        self.num_treatments = num_treatments
        self.max_group_size = max_group_size
        self.group_sizes = np.zeros(num_treatments)
    
    def assign_treatment(self, covariate, group, treatment):
        # Pocock-Simon minimization logic, adding constraint for group sizes
        eligible_treatments = [i for i in range(self.num_treatments) if self.group_sizes[i] < self.max_group_size]
        if not eligible_treatments:
            raise ValueError("All groups are full.")

        # Minimize discrepancy for eligible treatments
        assigned_treatment = random.choice(eligible_treatments)  # Replace this with minimization logic
        
        # Update the group sizes
        self.group_sizes[assigned_treatment] += 1
        return assigned_treatment

def Pocock_Simon_minimization(covariates, num_treatments):
    treatment_assignment = np.zeros(covariates.shape[0], dtype=int)
    index_shuffle = np.arange(covariates.shape[0])
    random.shuffle(index_shuffle)
    
    # Set the maximum group size for balanced groups
    max_group_size = covariates.shape[0] // num_treatments
    
    # Initialize treatment assignments for the first few participants
    ini_treatment = np.arange(num_treatments)
    treatment_assignment[index_shuffle[:num_treatments]] = ini_treatment
    
    # Create the minimization algorithm instance with group size enforcement
    minimization = MinimizationAlgorithm(num_treatments=num_treatments, max_group_size=max_group_size)
    
    # Update the group sizes for the initial assignments
    minimization.group_sizes += 1  # As each treatment got 1 participant
    
    # Start assigning the rest of the participants
    for i in range(num_treatments, covariates.shape[0]):
        # Subset of participants assigned so far
        treatment = treatment_assignment[index_shuffle[:i]]
        group = covariates[index_shuffle[:i], :]
        
        # Assign the next participant, enforcing group size balancing
        assigned_treatment = minimization.assign_treatment(covariates[index_shuffle[i], :], group, treatment)
        treatment_assignment[index_shuffle[i]] = assigned_treatment
    
    return treatment_assignment