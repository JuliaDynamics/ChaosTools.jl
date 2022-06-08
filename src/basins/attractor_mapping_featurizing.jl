export basins_fractions_clustering, basins_fractions
using Statistics: mean
using Neighborhood
using Distances, Clustering, Distributions
using ProgressMeter


#####################################################################################
# AttractorMapper API
#####################################################################################
struct AttractorsViaFeaturizing{DS<:GeneralizedDynamicalSystem, T, F, A, K, M} <: AttractorMapper
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
    rescale_features::Bool
    optimal_radius_method::String
end
DynamicalSystemsBase.get_rule_for_print(m::AttractorsViaFeaturizing) =
get_rule_for_print(m.ds)

function Base.show(io::IO, mapper::AttractorsViaFeaturizing)
    ps = generic_mapper_print(io, mapper)
    println(io, rpad(" type: ", ps), nameof(typeof(mapper.ds)))
    println(io, rpad(" featurizer: ", ps), DynamicalSystemsBase.eomstring(mapper.featurizer))
    println(io, rpad(" Ttr: ", ps), mapper.Ttr)
    println(io, rpad(" Δt: ", ps), mapper.Δt)
    println(io, rpad(" T: ", ps), mapper.total)
    println(io, rpad(" supervised: ", ps), !isnothing(mapper.attractors_ic))
    return
end

"""
Featurize the dataset and then cluster it via DBSCAN or kNN. If `dataset` is a Dictionary,
we consider it to be trajectories (including points in attractors) that are to be clustered.
In this case, we simply extract the features from them and cluster the features. If
`dataset` is an AbstractDataset or a function, then we consider it to be initial conditions
that are to be integrated, then featurized and then clustered.
"""
function cluster_datasets(dataset::Union{Dict, AbstractDataset, Function},
     mapper::AttractorsViaFeaturizing; show_progress=true, N=1000)
    if dataset isa Dict  #attractors
        feature_array = extract_features(mapper, dataset)
    elseif dataset isa Union{AbstractDataset, Function} #ics
        feature_array = extract_features(mapper, dataset; show_progress, N)
    else
        error("Incorrect input.")
    end
    class_labels, class_errors = classify_features(feature_array, mapper)
    return class_labels, class_errors
end

"""
    AttractorsViaFeaturizing(ds::DynamicalSystem, featurizer::Function; kwargs...) → mapper

Initialize a `mapper` that maps initial conditions to attractors using the featurizing and
clustering method of [^Stender2021]. See [`AttractorMapper`](@ref) for how to use the
`mapper`.

`featurizer` is a function that takes as an input an integrated trajectory `A::Dataset` and
the corresponding time vector `t` and returns a `Vector{<:Real}` of features describing the
trajectory.

## Keyword arguments
### Integration
* `T=100, Ttr=100, Δt=1, diffeq=NamedTuple()`: Propagated to [`trajectory`](@ref).

### Feature extraction and classification
* `attractors_ic = nothing` Enables supervised version, see below. If given, must be a
  `Dataset` of initial conditions each leading to a different attractor.
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
The trajectory `X` of each initial condition is transformed into a vector of features. Each
feature is a number useful in _characterizing the attractor_ the initial condition ends up
at, and distinguishing it from other attrators. Example features are the mean or standard
deviation of one of the of the timeseries of the trajectory, the entropy of the first two
dimensions, the fractal dimension of `X`, or anything else you may fancy. The vectors of
features are then used to identify to which attractor each trajectory belongs (i.e. in which
basin of attractor each initial condition is in). The method thus relies on the user having
at least some basic idea about what attractors to expect in order to pick the right
features, in contrast to [`AttractorsViaRecurrences`](@ref).

The algorithm of[^Stender2021] that we use has two versions to do this. If the attractors
are not known a-priori the **unsupervised versions** should be used. Here, the vectors of
features of each initial condition are mapped to an attractor by analysing how the features
are clustered in the feature space. Using the DBSCAN algorithm, we identify these clusters
of features, and consider each cluster to represent an attractor. Features whose attractor
is not identified are labeled as `-1`. If each feature spans different scales of magnitude,
rescaling them into the same `[0,1]` interval can bring significant improvements in the
clustering in case the `Euclidean` distance metric is used.   

In the **supervised version**, the attractors are known to the user, who provides one
initial condition for each attractor using the `attractors_ic` keyword. The algorithm then
evolves these initial conditions, extracts their features, and uses them as templates
representing the attrators. Each trajectory is considered to belong to the nearest template
(based on the distance in feature space). Notice that the functionality of this version is
similar to [`AttractorsViaProximity`](@ref). Generally speaking, the
[`AttractorsViaProximity`](@ref) is superior. However, if the dynamical system has extremely
high-dimensionality, there may be reasons to use the supervised method of this featurizing
algorithm instead.

## Parallelization note
The trajectories in this method are integrated in parallel using `Threads`. To enable this,
simply start Julia with the number of threads you want to use.

[^Stender2021]: Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
    stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function AttractorsViaFeaturizing(ds::GeneralizedDynamicalSystem, featurizer::Function;
        attractors_ic::Union{AbstractDataset, Nothing}=nothing, T=100, Ttr=100, Δt=1,
        clust_method_norm=Euclidean(),
        clustering_threshold = 0.0, min_neighbors = 10, diffeq = NamedTuple(),
        clust_method = clustering_threshold > 0 ? "kNN_thresholded" : "kNN", 
        rescale_features=true, optimal_radius_method="silhouettes",
    )
    if ds isa ContinuousDynamicalSystem
        T, Ttr, Δt = float.((T, Ttr, Δt))
    end
    return AttractorsViaFeaturizing(
        ds, Ttr, Δt, T, featurizer, attractors_ic, diffeq,
        clust_method_norm, clust_method, Float64(clustering_threshold), min_neighbors,
        rescale_features, optimal_radius_method
    )
end

# We need to extend the general `basins_fractions`, because the clustering method
# cannot map individual initial conditions to attractors
"""
`N = 1000` : number of initial conditions to be sampled, if `ics` is passed as a `Function`.
"""
function basins_fractions(mapper::AttractorsViaFeaturizing, ics::Union{AbstractDataset, Function};
        show_progress = true, N = 1000
    )
    class_labels,  = cluster_datasets(ics, mapper; show_progress, N)
    fs = basins_fractions(class_labels) # Vanilla fractions method with Array input
    if typeof(ics) <: AbstractDataset
        attractors = extract_attractors(mapper, class_labels, ics)
        return fs, class_labels, attractors
    else
        return fs
    end
end

function extract_features(mapper::AttractorsViaFeaturizing, ics::Union{AbstractDataset, Function};
    show_progress = true, N = 1000)

    N = (typeof(ics) <: Function)  ? N : size(ics, 1) # number of actual ICs

    feature_array = Vector{Vector{Float64}}(undef, N)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    Threads.@threads for i ∈ 1:N
        ic = _get_ic(ics,i)
        feature_array[i] = extract_features(mapper, ic)
        show_progress && ProgressMeter.next!(progress)
    end
    return reduce(hcat, feature_array) # Convert to Matrix from Vector{Vector}
end

function extract_features(mapper::AttractorsViaFeaturizing, u0::AbstractVector{<:Real})
    A = trajectory(mapper.ds, mapper.total, u0;
        Ttr = mapper.Ttr, Δt = mapper.Δt, diffeq = mapper.diffeq)
    t = (mapper.Ttr):(mapper.Δt):(mapper.total+mapper.Ttr)
    feature = mapper.featurizer(A, t)
    return feature
end

function extract_attractors(mapper::AttractorsViaFeaturizing, labels, ics)
    uidxs = unique(i -> labels[i], 1:length(labels))
    return Dict(labels[i] => trajectory(mapper.ds, mapper.total, ics[i];
    Ttr = mapper.Ttr, Δt = mapper.Δt, diffeq = mapper.diffeq) for i in uidxs if i ≠ -1)
end

"""
Extracts features from trajectories directly, including eg attractors that were identified
via XX. Receives a dictionary, with each entry being a vector of points whose features are
to be extracted.
"""
function extract_features(mapper::AttractorsViaFeaturizing, atts::Dict)

    N = length(atts) # number of attractors
    feature_array = Vector{Vector{Float64}}(undef, N)
    for i ∈ 1:N
        feature_array[i] = mapper.featurizer(atts[i], []) #t not being considered for featurizers, possible todo is to allow for that
    end
    return reduce(hcat, feature_array) # Convert to Matrix from Vector{Vector}
end

#####################################################################################
# Clustering classification low level code
#####################################################################################
function classify_features(features, mapper::AttractorsViaFeaturizing)
    if !isnothing(mapper.attractors_ic)
        classify_features_distances(features, mapper)
    else
        classify_features_clustering(features, mapper.min_neighbors, mapper.clust_method_norm,
        mapper.rescale_features, mapper.optimal_radius_method)
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

"""
Does "min-max" rescaling of vector `vec`: rescales it such that its values span `[0,1]`.
"""
function rescale(vec::Vector{T}) where T
    vec .-= minimum(vec)
    max = maximum(vec)
    if max == 0 return zeros(T, length(vec)) end
    vec ./= maximum(vec)
end

# Unsupervised method: clustering in feature space
function classify_features_clustering(features, min_neighbors, metric, rescale_features,
     optimal_radius_method)
    dimfeats, nfeats = size(features); if dimfeats ≥ nfeats return 1:nfeats, zeros(nfeats) end  #needed because dbscan, as implemented, needs to receive as input a matrix D x N such that D < N
    if rescale_features features = mapslices(rescale, features, dims=2) end
    ϵ_optimal = optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method)
    # Now recalculate the final clustering with the optimal ϵ
    clusters = dbscan(features, ϵ_optimal; min_neighbors)
    clusters, sizes = sort_clusters_calc_size(clusters)
    class_labels = cluster_assignment(clusters, features; include_boundary=false)
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

