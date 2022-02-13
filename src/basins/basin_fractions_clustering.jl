export basin_fractions_clustering, basin_fractions
using Statistics: mean
using Neighborhood #for kNN
using Distances, Clustering, Distributions
using ProgressMeter


#####################################################################################
# AttractorMapper API
#####################################################################################
struct AttractorsViaFeaturizing{DS<:DynamicalSystem, T, F, A, K, M} <: AttractorMapper
    ds::DS
    Ttr::T
    Δt::T
    total::T
    featurizer::F
    attractors_ic::A
    diffeq::K
    clust_method_norm::M
    clust_method::String
    clustering_threshold::Float64
    min_neighbors::Int
end


"""
    AttractorsViaFeaturizing(ds::DynamicalSystem, featurizer::Function; kwargs...) → mapper

Initialize a `mapper` to be used with [`basin_fractions`](@ref) that maps initial conditions
to attractors using the featurizing and clustering method of[^Stender2021].

`featurizer` is a function that takes as an input an integrated trajectory `A::Dataset`
and the corresponding time vector `t` and returns a `Vector{<:Real}` of features
describing the trajectory. 
The _optional_ argument `attractors_ic`, if given, must be a `Dataset` of initial conditions
each leading to a different attractor. Giving this argument switches the method from
unsupervised (default) to supervised, see description below.

## Keyword arguments
### Integration
* `T=100, Ttr=100, Δt=1, diffeq=NamedTuple()`: Propagated to [`trajectory`](@ref).

### Feature extraction and classification
* `attractors_ic = nothing` Enables supervised version, see below. 
* `min_neighbors = 10`: (unsupervised method only) minimum number of neighbors
  (i.e. of similar features) each feature needs to have in order to be considered in a
  cluster (fewer than this, it is labeled as an outlier, `-1`).
* `clust_method_norm=Euclidean()`: metric to be used in the clustering.
* `clustering_threshold = 0.0`: Maximum allowed distance between a feature and the
  cluster center for it to be considered inside the cluster.
  Only used when `clust_method = "kNN_thresholded"`.
* `clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN"`: 
  (supervised method only) which clusterization method to
  apply. If `"kNN"`, the first-neighbor clustering is used. If `"kNN_thresholded"`, a
  subsequent step is taken, which considers as unclassified (label `-1`) the features
  whose distance to the nearest template is above the `clustering_threshold`.

## Description
The trajectory `X` of each initial condition is transformed into a vector of features.
Each feature is a number useful in _characterizing the attractor_ the initial condition
ends up at, and distinguishing it from other attrators.
The vectors of features are then used to identify to which attractor
each trajectory belongs (i.e. in which basin of attractor each initial condition is in).
The method thus relies on the user having at least some basic idea about what attractors
to expect in order to pick the right features, in contrast to [`AttractorsViaRecurrences`](@ref).

The algorithm of[^Stender2021] that we use has two versions to do this.
If the attractors are not known a-priori the **unsupervised versions** should be used.
Here, the vectors of features of each initial condition are mapped to an attractor by
analysing how the features are clustered in the feature space. Using the DBSCAN algorithm,
we identify these clusters of features, and consider each cluster to represent an
attractor. Features whose attractor is not identified are labeled as `-1`.

In the **supervised version**, the attractors are known to the user, who provides one
initial condition for each attractor using the `attractors_ic` keyword.
The algorithm then evolves these initial conditions, extracts their features, and uses them
as templates representing the attrators. Each trajectory is considered to belong to the
nearest template (based on the distance in feature space).
Notice that the functionality of this version is similar to [`AttractorsViaProximity`](@ref).
Generally speaking, the [`AttractorsViaProximity`](@ref) is superior. However, if the
dynamical system has extremely high-dimensionality, there may be reasons to use the
supervised method of this featurizing algorithm instead.

## Parallelization note
The trajectories in this method are integrated in parallel using `Threads`.
To enable this, simply start Julia with the number of threads you want to use.

[^Stender2021]: Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function AttractorsViaFeaturizing(ds::DynamicalSystem, featurizer::Function;
        attractors_ic::Union{AbstractDataset, Nothing}=nothing, T=100, Ttr=100,Δt=1,
        clust_method_norm=Euclidean(), 
        clustering_threshold = 0.0, min_neighbors = 10, diffeq = NamedTuple(),
        clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN",
    )
    return AttractorsViaFeaturizing(
        ds, Ttr, Δt, T, featurizer, attractors_ic, diffeq,
        clust_method_norm, clust_method, clustering_threshold, min_neighbors
    )
end

function basin_fractions(mapper::AttractorsViaFeaturizing, ics::Union{AbstractDataset, Function};
        show_progress = true, N = 1000
    )
    feature_array = extract_features(mapper, ics; show_progress, N)
    class_labels, = classify_features(feature_array, mapper)
    fs = basin_fractions(class_labels) # Vanilla fractions method with Array input
    return typeof(ics) <: AbstractDataset ? (fs, class_labels) : fs
end

function extract_features(mapper::AttractorsViaFeaturizing, ics::Union{AbstractDataset, Function};
    show_progress = true, N = 1000)

    N = (typeof(ics) <: Function)  ? N : size(ics, 1) #number of actual ICs

    feature_array = Vector{Vector{Float64}}(undef, N)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    for i = 1:N
        ic = _get_ic(ics,i)
        feature_array[i] = extract_features(mapper, ic)
        show_progress && ProgressMeter.next!(progress)
    end
    return reduce(hcat, feature_array) # Convert to Matrix from Vector{Vector}
end

function extract_features(mapper::AttractorsViaFeaturizing, u0::AbstractVector{<:Real})
    u = trajectory(mapper.ds, mapper.total, u0; 
    Ttr=mapper.Ttr, Δt=mapper.Δt, diffeq=mapper.diffeq)
    t = (mapper.Ttr):(mapper.Δt):(mapper.total+mapper.Ttr)
    feature = mapper.featurizer(u, t)
    return feature
end

#####################################################################################
# Clustering classification low level code
#####################################################################################
function classify_features(features, mapper::AttractorsViaFeaturizing)
    if !isnothing(mapper.attractors_ic)
        classify_features_distances(features, mapper)
    else
        classify_features_clustering(features, mapper.min_neighbors)
    end
end

# Supervised method: closest attractor template in feature space
function classify_features_distances(features, mapper)

    templates = extract_features(mapper, mapper.attractors_ic; show_progress=false)

    if mapper.clust_method == "kNN" || mapper.clust_method == "kNN_thresholded"
        template_tree = searchstructure(KDTree, templates, mapper.clust_method_norm)
        class_labels, class_errors = bulksearch(template_tree, features, NeighborNumber(1))
        class_labels = reduce(vcat, class_labels) # make it a vector
        if mapper.clust_method == "kNN_thresholded" # Make label -1 if error bigger than threshold
            class_errors = reduce(vcat, class_errors)
            class_labels[class_errors .≥ mapper.clustering_threshold] .= -1
        end
    else
        error("Incorrect clustering mode.")
    end
    return class_labels, class_errors
end

# Unsupervised method: clustering in feature space
function classify_features_clustering(features, min_neighbors)
    ϵ_optimal = optimal_radius(features, min_neighbors)
    # Now recalculate the final clustering with the optimal ϵ
    clusters = Clustering.dbscan(features, ϵ_optimal; min_neighbors)
    clusters, sizes = sort_clusters_calc_size(clusters) 
    class_labels = cluster_props(clusters, features; include_boundary=false)
    # number of real clusters (size above minimum points);
    # this is also the number of "templates"
    k = length(sizes[sizes .> min_neighbors])
    # create templates/labels, assign errors
    class_errors = zeros(size(features)[2])
    for i=1:k
        idxs_cluster = class_labels .== i
        center = mean(features[:, class_labels .== i], dims=2)[:,1]
        dists = colwise(Euclidean(), center, features[:, idxs_cluster])
        class_errors[idxs_cluster] = dists
    end

    return class_labels, class_errors
end

#####################################################################################
# Utilities
#####################################################################################
"""
Util function for `classify_features`. It returns the size of all the DBSCAN clusters and the
assignment vector, in whch the i-th component is the cluster index of the i-th feature
"""
function cluster_props(clusters, data; include_boundary=true)
    assign = zeros(Int, size(data)[2])
    for (idx, cluster) in enumerate(clusters)
        assign[cluster.core_indices] .= idx
        if cluster.boundary_indices != []
            if include_boundary
                assign[cluster.boundary_indices] .= idx
            else
                assign[cluster.boundary_indices] .= -1
            end
        end
    end
    return assign
end

"""
Util function for `classify_features`. Calculates the clusters' (DbscanCluster) size 
and sorts them in decreasing order according to the size.
"""
function sort_clusters_calc_size(clusters)
    sizes = [cluster.size for cluster in clusters]
    idxsort = sortperm(sizes,rev=true)
    return clusters[idxsort], sizes[idxsort]
end

"""
Find the optimal radius ε of a point neighborhood for use in DBSCAN, in the unsupervised 
`classify_features`. It does so by finding the `ε` which maximizes the minimum silhouette
of the cluster.
"""
function optimal_radius(features, min_neighbors)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/200, minimum(feat_ranges), length=200)
    s_grid = zeros(size(ϵ_grid)) # min silhouette values (which we want to maximize)

    #vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i=1:length(ϵ_grid)
        clusters = dbscan(features, ϵ_grid[i], min_neighbors=min_neighbors)
        dists = pairwise(Euclidean(), features)
        class_labels = cluster_props(clusters, features)
        if length(clusters) ≠ 1 #silhouette undefined if only one cluster.
            sils = silhouettes(class_labels, dists) #values == 0 are due to boundary points
            s_grid[i] = minimum(sils[sils .!= 0.0]) #minimum silhouette value of core points
        else
            s_grid[i] = -2; #this would effecively ignore the single-cluster solution
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end