using Distances, Clustering, Distributions
export ClusterConfig, cluster_features

"""
    ClusteringConfig(; kwargs...)

Initialize a struct that contains information used to cluster "features".
These features are typically extracted from trajectories/datasets in
[`AttractorsViaFeaturizing`](@ref), or manualy by the user.

The clustering is done in the function [`cluster_features`](@ref).

The default clustering method is that used in [^Stender2021]
which is unsupervised, see Description below.

## Keyword arguments
* `templates = nothing`: Enables supervised version, see below. If given, must be a
  vector of `Vector`s, each inner vector containing the features representing a center of a
  cluster.
* `min_neighbors = 10`: (unsupervised method only) minimum number of neighbors (i.e. of
  similar features) each feature needs to have in order to be considered in a cluster (fewer
  than this, it is labeled as an outlier, `-1`).
* `clust_method_norm = Euclidean()`: metric to be used in the clustering.
* `clustering_threshold = 0.0`: Maximum allowed distance between a feature and the cluster
  center for it to be considered inside the cluster. Only used when `clust_method =
  "kNN_thresholded"`.
* `clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN"`: (supervised method
  only) which clusterization method to apply. If `"kNN"`, the first-neighbor clustering is
  used. If `"kNN_thresholded"`, a subsequent step is taken, which considers as unclassified
  (label `-1`) the features whose distance to the nearest template is above the
  `clustering_threshold`.
* `rescale_features = true`: (unsupervised method): if true, rescale each dimension of the
  extracted features separately into the range `[0,1]`. This typically leads to
  more accurate clustering.
* `optimal_radius_method = silhouettes` (unsupervised method): the method used to determine
  the optimal radius for clustering features in the unsupervised method. The `silhouettes`
  method chooses the radius that maximizes the average silhouette values of clusters, and
  is an iterative optimization procedure that may take some time to execute (see
  [`optimal_radius_dbscan_silhouettes`](@ref) for details). The `elbow`
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

There are two versions to do this. The **unsupervised version** does not rely on templates,
and instead uses the DBSCAN clustering algorithm to identify clusters of similar features.
To achieve this, each feature is considered a point in feature space. In this space, the
algorithm basically groups points that are closely packed. To achieve this, a crucial parameter
is a radius for  distance `ϵ` that determines the "closeness" of points in clusters.
Two methods are currently implemented to determine an `optimal_radius`, as described and
referred in `optimal_radius_method` above.

Features whose cluster is not identified are labeled as `-1`.

In the **supervised version**, the user provides features to be used as templates guiding the
clustering via the `templates` keyword. Each feature is considered to belong to
the "cluster" of the nearest template (based on the distance in feature space).

[^Stender2021]:
    Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
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

function ClusteringConfig(; templates::Union{Nothing, Dict} = nothing,
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
"""
    cluster_features(features, cluster_specs::ClusteringConfig)
Cluster the given `features::Vector{<:AbstractVector}`,
according to given [`ClusteringConfig`](@ref).
Return `cluster_labels, cluster_errors`, which respectively contain, for each feature, the labels
(indices) of the corresponding cluster and the error associated with that clustering.
The error is the distance from the feature to (i) the cluster, in the supervised method
or (ii) to the center of the cluster, in the unsupervised method.
"""
function cluster_features(features::Vector{<:AbstractVector}, cluster_specs::ClusteringConfig)
    # All methods require the features in a matrix format
    f = reduce(hcat, features) # Convert to Matrix from Vector{Vector}
    f = float.(f)
    if !isnothing(cluster_specs.templates)
        cluster_features_distances(f, cluster_specs)
    else
        cluster_features_clustering(
            f, cluster_specs.min_neighbors, cluster_specs.clust_method_norm,
            cluster_specs.rescale_features, cluster_specs.optimal_radius_method
        )
    end
end

# Supervised method: closest attractor template in feature space
function cluster_features_distances(features, cluster_specs)
    templates = float.(reduce(hcat, [cluster_specs.templates[i] for i ∈ keys(cluster_specs.templates)] ) ) #puts each vector into a column, with the ordering based on the order given in keys(d)
    if !(cluster_specs.clust_method == "kNN" ||
         cluster_specs.clust_method == "kNN_thresholded")
        error("Incorrect clustering mode for \"supervised\" method.")
    end
    template_tree = searchstructure(KDTree, templates, cluster_specs.clust_method_norm)
    cluster_labels, cluster_errors = bulksearch(template_tree, features, NeighborNumber(1))
    cluster_labels = reduce(vcat, cluster_labels) # make it a vector
    cluster_errors = reduce(vcat, cluster_errors)
    if cluster_specs.clust_method == "kNN_thresholded"
        # Make label -1 if error bigger than threshold
        cluster_labels[cluster_errors .≥ cluster_specs.clustering_threshold] .= -1
    end
    matlabels_to_dictlabels = Dict(1:length(keys(cluster_specs.templates)).=>keys(cluster_specs.templates))
    cluster_user_labels = replace(cluster_labels, matlabels_to_dictlabels...) #cluster_user_labels[i] is the label of template nearest to feature i (i-th column of features matrix)
    return cluster_user_labels, cluster_errors
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

    if rescale_features
        features = mapslices(_rescale!, features; dims=2)
    end
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
        center = mean(features[:, idxs_cluster], dims=2)[:, 1]
        dists = colwise(Euclidean(), center, features[:, idxs_cluster])
        cluster_errors[idxs_cluster] = dists
    end

    return cluster_labels, cluster_errors
end