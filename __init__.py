# Import everything from the whomp module

from .whomp import (
    KMeansConstrained,
    k_means_constrained,
    kmeans_constrained_single,
    _labels_constrained,
    solve_optimal_transport,
    RandomPartition,
    Barycenter_Fixed_Point_LP,
    WHOMP_Matching,
    WHOMP_Random,
    MinimizationAlgorithm,
    Pocock_Simon_minimization
)

# Set the version of the WHOMP module
__version__ = "0.1.0"

# Define the public API for the module
__all__ = [
    "KMeansConstrained",
    "k_means_constrained",
    "kmeans_constrained_single",
    "_labels_constrained",
    "solve_optimal_transport",
    "RandomPartition",
    "Barycenter_Fixed_Point_LP",
    "WHOMP_Matching",
    "WHOMP_Random",
    "MinimizationAlgorithm",
    "Pocock_Simon_minimization"
]