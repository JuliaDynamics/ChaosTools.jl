using Distances, Clustering, Distributions
export ClusterConfig
export cluster_datasets, cluster_features

# TODO: This docstirng needs to be reworked completely
"""
    ClusteringConfig(; kwargs...)

Initialize a struct that contains information used to cluster "features".
These features are typically extracted from trajectories/datasets in
[`AttractorsViaFeaturizing`](@ref), or manualy by the user.

The clustering is done in the function [`cluster_features`](@ref).

## Keyword arguments
* `templates = nothing` Enables supervised version, see below. If given, must be a
  `Dataset` of initial conditions each leading to a different attractor, to which
  trajectories will be matched.
* `min_neighbors = 10`: (unsupervised method only) minimum number of neighbors (i.e. of
  similar features) each feature needs to have in order to be considered in a cluster (fewer
  than this, it is labeled as an outlier, `-1`).
* `clust_method_norm=Euclidean()`: metric to be used in the clustering.
* `clustering_threshold = 0.0`: Maximum allowed distance between a feature and the cluster
  center for it to be considered inside the cluster. Only used when `clust_method =
  "kNN_thresholded"`.
* `clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN"`: (supervised method
  only) which clusterization method to apply. If `"kNN"`, the first-neighbor clustering is
  used. If `"kNN_thresholded"`, a subsequent step is taken, which considers as unclassified
  (label `-1`) the features whose distance to the nearest template is above the
  `clustering_threshold`.
* `rescale_features = true`: (unsupervised method): if true, rescale each dimension of the
extracted features separately into the range `[0,1]`.
* `optimal_radius_method = silhouettes` (unsupervised method): the method used to determine
the optimal radius for clustering features in the unsupervised method. The `silhouettes`
    method chooses the radius that maximizes the average silhouette values of clusters, and
    is an iterative optimization procedure that may take some time to execute. The `elbow`
    method chooses the the radius according to the elbow (knee, highest-derivative method)
    (see [`optimal_radius_dbscan_elbow`](@ref) for details), and is quicker though possibly
    leads to worse clustering.

## Description
The trajectory `X`, which may for instance be an attractor, is transformed into a vector of
features. Each feature is a number useful in _characterizing the attractor_, and
distinguishing it from other attrators. Example features are the mean or standard deviation
of one of the of the timeseries of the trajectory, the entropy of the first two dimensions,
the fractal dimension of `X`, or anything else you may fancy. The vectors of features are
then used to identify clusters of attractors.

There are two versions to do this. The **unsupervised versions** will cluster trajectories
(e.g. attractors) using the DBSCAN algorithm, which does not rely on templates. Features whose
cluster is not identified are labeled as `-1`. If each attractors spans different scales of
magnitude, rescaling them into the same `[0,1]` interval can bring significant improvements
in the clustering in case the `Euclidean` distance metric is used.

In the **supervised version**, the user provides features to be used as templates guiding the
clustering via the `templates` keyword. Each trajectory is considered to belong to
the nearest template (based on the distance in feature space).

[^Stender2021]: Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
    stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
mutable struct ClusteringConfig{A, M}
    templates::A
    clust_method_norm::M
    clust_method::String
    clustering_threshold::Float64
    min_neighbors::Int
    rescale_features::Bool
    optimal_radius_method::String
end

function ClusteringConfig(; templates::Union{Nothing, Vector} = nothing,
        clust_method_norm=Euclidean(), clustering_threshold = 0.0, min_neighbors = 10,
        clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN",
        rescale_features=true, optimal_radius_method="silhouettes",
    )
    return ClusteringConfig(
        templates, clust_method_norm, clust_method,
        Float64(clustering_threshold), min_neighbors,
        rescale_features, optimal_radius_method
    )
end

include("cluster_utils.jl")

#####################################################################################
# Clustering classification functions
#####################################################################################
# TODO: Wait, do we need this function...?
function cluster_datasets(dataset, featurizer, cluster_specs::ClusteringConfig)
    feature_array = extract_features(featurizer, dataset)
    return cluster_features(feature_array, cluster_specs)
end

"""
    cluster_features(features, cluster_specs::ClusteringConfig)
Cluster the given `features::Vector{<:AbstractVector}`,
according to given [`ClusteringConfig`](@ref).
Return `cluster_labels, cluster_errors`, which are
"""
function cluster_features(features::Vector{<:AbstractVector}, cluster_specs::ClusteringConfig)
    if !isnothing(cluster_specs.templates)
        cluster_features_distances(features, cluster_specs)
    else
        cluster_features_clustering(
            features, cluster_specs.min_neighbors, cluster_specs.clust_method_norm,
            cluster_specs.rescale_features, cluster_specs.optimal_radius_method
        )
    end
end

# Supervised method: closest attractor template in feature space
function cluster_features_distances(features, cluster_specs)
    # casting to floats needed for kNN
    templates = float.(reduce(hcat, cluster_specs.templates))

    if !(cluster_specs.clust_method == "kNN" ||
        cluster_specs.clust_method == "kNN_thresholded")
        error("Incorrect clustering mode for \"supervised\" method.")
    end
    template_tree = searchstructure(KDTree, templates, cluster_specs.clust_method_norm)
    cluster_labels, cluster_errors = bulksearch(template_tree, features, NeighborNumber(1))
    cluster_labels = reduce(vcat, cluster_labels) # make it a vector
    if cluster_specs.clust_method == "kNN_thresholded"
        # Make label -1 if error bigger than threshold
        cluster_errors = reduce(vcat, cluster_errors)
        cluster_labels[cluster_errors .≥ cluster_specs.clustering_threshold] .= -1
    end
    return cluster_labels, cluster_errors
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
        features, min_neighbors, metric, rescale_features, optimal_radius_method
    )
    # needed because dbscan, as implemented, needs to receive as input a matrix D x N
    # such that D < N
    dimfeats, nfeats = size(features)
    if dimfeats ≥ nfeats return 1:nfeats, zeros(nfeats) end

    if rescale_features; features = mapslices(_rescale!, features; dims=2); end
    ϵ_optimal = optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method)
    # Now recalculate the final clustering with the optimal ϵ
    clusters = dbscan(features, ϵ_optimal; min_neighbors)
    clusters, sizes = sort_clusters_calc_size(clusters)
    cluster_labels = cluster_assignment(clusters, features; include_boundary=false)
    # number of real clusters (size above minimum points);
    # this is also the number of "templates"
    k = length(sizes[sizes .> min_neighbors])
    # create templates/labels, assign errors
    cluster_errors = zeros(size(features)[2])
    for i=1:k
        idxs_cluster = cluster_labels .== i
        center = mean(features[:, cluster_labels .== i], dims=2)[:,1]
        dists = colwise(Euclidean(), center, features[:, idxs_cluster])
        cluster_errors[idxs_cluster] = dists
    end

    return cluster_labels, cluster_errors
end