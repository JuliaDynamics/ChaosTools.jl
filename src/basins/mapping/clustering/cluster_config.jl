using Distances, Clustering, Distributions
export ClusteringConfig, cluster_features

"""
    ClusteringConfig(; kwargs...)

Initialize a struct that contains information used to cluster "features".
These features are typically extracted from trajectories/datasets in
[`AttractorsViaFeaturizing`](@ref), or manualy by the user.

The clustering is done in the function [`cluster_features`](@ref).

The default clustering method is an improvement over existing literature, see Description.

## Keyword arguments
* `templates = nothing`: Enables supervised version, see below. If given, must be a
  `Dict` of cluster labels to cluster features. The labels must be of `Int` type,
  and the features are `Vector`s representing a cluster (which can be an attractor, for
  instance). The label `-1` is reserved for invalid trajectories, which either diverge or
  whose clustering failed.
### Supervised method
* `clustering_threshold = 0.0`: Maximum allowed distance between a feature and the cluster
  center for it to be considered inside the cluster. Only used when `clust_method =
  "kNN_thresholded"`.
* `clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN"`: (supervised method
  only) which clusterization method to apply. If `"kNN"`, the first-neighbor clustering is
  used. If `"kNN_thresholded"`, a subsequent step is taken, which considers as unclassified
  (label `-1`) the features whose distance to the nearest template is above the
  `clustering_threshold`.
### Unsupervised method
* `rescale_features = true`: if true, rescale each dimension of the
  extracted features separately into the range `[0,1]`. This typically leads to
  more accurate clustering.
* `min_neighbors = 10`: (unsupervised method only) minimum number of neighbors (i.e. of
  similar features) each feature needs to have in order to be considered in a cluster (fewer
  than this, it is labeled as an outlier, `-1`).
* `clust_method_norm = Euclidean()`: metric to be used in the clustering.
* `optimal_radius_method::String = "silhouettes_optim"`: the method used to
  determine the optimal radius for clustering features in the unsupervised method.
  Possible values are:
  - `"silhouettes"`: Performs a linear (sequential) search for the radius that maximizes a
    statistic of the silhouette values of clusters (typically the mean). This can be chosen
    with `silhouette_statistic`. The linear search may take some time to finish. To
    increase speed, the number of radii iterated through can be reduced by decreasing
    `num_attempts_radius` (see its entry below).
  - `"silhouettes_optim"`: Same as `"silhouettes"` but performs an optimized search via
    Optim.jl. It's faster than `"silhouettes"`, with typically the same accuracy (the
    search here is not guaranteed to always find the global maximum, though it typically
    gets close).
  - `"knee"`: chooses the the radius according to the knee (a.k.a. elbow,
    highest-derivative method) and is quicker, though generally leading to much worse
    clustering. It requires that `min_neighbors` > 1.
### Keywords for optimal radius estimation
* `num_attempts_radius = 100`: number of radii that
  the `optimal_radius_method` will try out in its iterative procedure. Higher values
  increase the accuracy of clustering, though not necessarily much, while always reducing
  speed.
* `silhouette_statistic::Function = mean`: statistic
  (e.g. mean or minimum) of the silhouettes that is maximized in the "optimal" clustering.
  The original implementation in [^Stender2021] used the `minimum` of the silhouettes, and
  typically performs less accurately than the `mean`.
* `max_used_features = 0`: if not `0`, it should be an `Int` denoting the max
  amount of features to be used when finding the optimal radius. Useful when clustering
  a very large number of features (e.g., high accuracy estimation of fractions of basins
  of attraction).

## Description
The trajectory `X`, which may for instance be an attractor, is transformed into a vector
of features. Each feature is a number useful in _characterizing the attractor_, and
distinguishing it from other attrators. Example features are the mean or standard
deviation of one of the of the timeseries of the trajectory, the entropy of the first two
dimensions, the fractal dimension of `X`, or anything else you may fancy. The vectors of
features are then used to identify clusters of attractors.

There are two versions to do this. The **unsupervised version** does not rely on
templates, and instead uses the DBSCAN clustering algorithm to identify clusters of
similar features. To achieve this, each feature is considered a point in feature space. In
this space, the algorithm basically groups points that are closely packed. To achieve
this, a crucial parameter is a radius for  distance `ϵ` that determines the "closeness" of
points in clusters.

Two methods are currently implemented to determine an `optimal_radius`, as described in
`optimal_radius_method` above. The default method is based on the silhouette values of
clusters. A silhouette value measures how similar a point is to the cluster it currently
belongs, compared to the other clusters, and ranges from -1 (worst matching) to +1 (ideal
matching). If only one cluster is found, the assigned silhouette is 0. The default method
,`silhouettes`, chooses the radius `ε` that maximizes the average silhouette across all
clusters.  The alternative `elbow` method works by calculating the distance of each point
to its k-nearest-neighbors (with `k=min_neighbors`) and finding the distance corresponding
to the highest derivative in the curve of the distances, sorted in ascending order. This
distance is chosen as the optimal radius. It is described in [^Kriegel1996] and
[^Schubert2017].

In the **supervised version**, the user provides features to be used as templates guiding
the clustering via the `templates` keyword. Each feature is considered to belong to the
"cluster" of the nearest template (based on the distance in feature space), and is
labelled following the template's label, given in `templates`.

In both versions, features whose cluster is not identified are labeled as `-1`.

[^Stender2021]:
    Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
    stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
[^Kriegel1996]: Ester, Kriegel, Sander and Xu: A Density-Based Algorithm for Discovering
    Clusters in Large Spatial Databases with Noise
[^Schubert2017]:
    Schubert, Sander, Ester, Kriegel and Xu: DBSCAN Revisited, Revisited: Why and How You
    Should (Still) Use DBSCAN
"""
mutable struct ClusteringConfig{A, M}
    templates::A
    clust_method_norm::M
    clust_method::String
    clustering_threshold::Float64
    min_neighbors::Int
    rescale_features::Bool
    optimal_radius_method::String
    num_attempts_radius::Int
    silhouette_statistic::Function
    max_used_features::Int
end

function ClusteringConfig(; templates::Union{Nothing, Dict} = nothing,
        clust_method_norm=Euclidean(), clustering_threshold = 0.0, min_neighbors = 10,
        clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN",
        rescale_features=true, optimal_radius_method="silhouettes_optim",
        num_attempts_radius=100, silhouette_statistic = mean, max_used_features = 0,
    )
    return ClusteringConfig(
        templates, clust_method_norm, clust_method,
        Float64(clustering_threshold), min_neighbors,
        rescale_features, optimal_radius_method,
        num_attempts_radius, silhouette_statistic, max_used_features
    )
end

include("cluster_utils.jl")

#####################################################################################
# Clustering classification functions
#####################################################################################
"""
    cluster_features(features, cc::ClusteringConfig)
Cluster the given `features::Vector{<:AbstractVector}`,
according to given [`ClusteringConfig`](@ref).
Return `cluster_labels, cluster_errors`, which respectively contain, for each feature, the labels
(indices) of the corresponding cluster and the error associated with that clustering.
The error is the distance from the feature to (i) the cluster, in the supervised method
or (ii) to the center of the cluster, in the unsupervised method.
"""
function cluster_features(features::Vector{<:AbstractVector}, cc::ClusteringConfig)
    # All methods require the features in a matrix format
    f = reduce(hcat, features) # Convert to Matrix from Vector{Vector}
    f = float.(f)
    if !isnothing(cc.templates)
        cluster_features_distances(f, cc)
    else
        cluster_features_clustering(
            f, cc.min_neighbors, cc.clust_method_norm,
            cc.rescale_features, cc.optimal_radius_method,
            cc.num_attempts_radius, cc.silhouette_statistic, cc.max_used_features,
        )
    end
end

# Supervised method: closest attractor template in feature space
function cluster_features_distances(features, cc::ClusteringConfig)
    #puts each vector into a column, with the ordering based on the order given in keys(d)
    templates = float.(reduce(hcat, [cc.templates[i] for i ∈ keys(cc.templates)]))
    if !(cc.clust_method == "kNN" ||
         cc.clust_method == "kNN_thresholded")
        error("Incorrect clustering mode for \"supervised\" method.")
    end
    template_tree = searchstructure(KDTree, templates, cc.clust_method_norm)
    cluster_labels, cluster_errors = bulksearch(template_tree, features, NeighborNumber(1))
    cluster_labels = reduce(vcat, cluster_labels) # make it a vector
    if cc.clust_method == "kNN_thresholded"
        cluster_errors = reduce(vcat, cluster_errors)
        # Make label -1 if error bigger than threshold
        cluster_labels[cluster_errors .≥ cc.clustering_threshold] .= -1
    end
    matlabels_to_dictlabels = Dict(1:length(keys(cc.templates)) .=> keys(cc.templates))
    # cluster_user_labels[i] is the label of template nearest to feature i
    # (i-th column of features matrix)
    cluster_user_labels = replace(cluster_labels, matlabels_to_dictlabels...)
    return cluster_user_labels
end

"""
Do "min-max" rescaling of vector `vec`: rescale it such that its values span `[0,1]`.
"""
function _rescale!(vec::Vector{T}) where T
    vec .-= minimum(vec)
    max = maximum(vec)
    if max == 0 return zeros(T, length(vec)) end
    vec ./= maximum(vec)
end

# Unsupervised method: clustering in feature space
function cluster_features_clustering(
        features, min_neighbors, metric, rescale_features, optimal_radius_method,
        num_attempts_radius, silhouette_statistic, max_used_features
    )
    # needed because dbscan, as implemented, needs to receive as input a matrix D x N
    # such that D < N
    dimfeats, nfeats = size(features)
    if dimfeats ≥ nfeats @warn "Not enough features. The algorithm needs the number of features
         $nfeats to be greater or equal than the number of dimensions $dimfeats";
           return 1:nfeats, zeros(nfeats) end

    if rescale_features
        features = mapslices(_rescale!, features; dims=2)
    end
    # These functions are called from cluster_utils.jl
    features_for_optimal = if max_used_features == 0
        features
    else
        StatsBase.sample(features, minimum(length(features), max_used_features); replace = false)
    end
    ϵ_optimal = optimal_radius_dbscan(
        features_for_optimal, min_neighbors, metric, optimal_radius_method,
        num_attempts_radius, silhouette_statistic
    )
    dists = pairwise(metric, features)
    dbscanresult = dbscan(dists, ϵ_optimal, min_neighbors)
    cluster_labels = cluster_assignment(dbscanresult)
    return cluster_labels
end
