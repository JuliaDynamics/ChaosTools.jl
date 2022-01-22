export basin_fractions_clustering, basin_fractions
using Statistics: mean
using Neighborhood #for kNN
using Distances, Clustering, Distributions
using ProgressMeter

"""
    basin_fractions_clustering(
        ds::DynamicalSystem, featurizer::Function,
        ics::Union{Dataset, Function} [, attractors_ic]; kwargs...
    ) → fs  OR  (fs, labels)

Compute the state space fractions `fs` of the basins of attraction of the given dynamical
system using the random sampling & clustering method of [^Stender2021].

`featurizer` is a function that takes as an input an integrated trajectory `A::Dataset`
and the corresponding time vector `t` and returns a vector `Vector{<:Real}` of features
describing the trajectory. Initial conditions are sampled from `ics`, which can either
be a `Dataset` of initial conditions, or a 0-argument function `ics()` that spits out
random initial conditions. See [`statespace_sampler`](@ref) to generate such functions.

The output `fs` is a dictionary whose keys are the labels given to each attractor, and the 
values are their respective fractions. The label `-1` is given to any initial condition
whose attractor did not match any of the clusters, see description below.

If `ics` is a `Dataset`, besides `fs` the `labels` of each initial condition are also
returned.

The _optional_ argument `attractors_ic`, if given, must be a `Dataset` of initial conditions
each leading to a different attractor. Giving this argument switches the method from
unsupervised (default) to supervised, see description below.

## Keyword arguments
### Integration
* `T=100, Ttr=100, Δt=1, diffeq=NamedTuple()`: Propagated to [`trajectory`](@ref). 
* `N=1000`: Number of sample initial conditions to generate in case `ics` is a function.
* `show_progress = true`: Display a progress bar of the process.

### Feature extraction and classification
* `clust_method_norm=Euclidean()`: metric to be used in the clustering.
* `clust_method = "kNN"`: (supervised method only) which clusterization method to
  apply. If `"kNN"`, the first-neighbor clustering is used. If `"kNN_thresholded"`, a
  subsequent step is taken, which considers as unclassified (label `-1`) the features
  whose distance to the nearest template above the `clustering_threshold`.
* `clustering_threshold = 0.0`: Maximum allowed distance between a feature and the 
  cluster center for it to be considered inside the cluster.
  Only used when `clust_method = "kNN_thresholded"`.
* `min_neighbors = 10`: (unsupervised method only) minimum number of neighbors
  (i.e. of similar features) each feature needs to have in order to be considered in a
  cluster (fewer than this, it is labeled as an outlier, id=-1). This number is somewhat hard
  to define, as it directly interferes with how many attractors the clustering finds.

## Description
Let ``F(A)`` be the fraction of initial conditions in a region of state space
``\\mathcal{S}`` (represented by `ics`) which are in the basin of attraction of an attractor
``A``. `basin_fractions_clustering` estimates ``F`` for attractors in
``\\mathcal{S}`` by counting which initial conditions end up in which attractors.

The trajectory `X` of each initial condition is transformed into a vector of features. 
Each feature is a number useful in *characterizing the attractor* and distinguishing it
from other attrators. The vectors of features are then used to identify to which attractor
each trajectory belongs (i.e. in which basin of attractor each initial condition is in).
The method thus relies on the user having at least some basic idea about what attractors
to expect in order to pick the right features, in contrast to [`basins_of_attraction`](@ref).

The algorithm of[^Stender2021] that we use has two methods to do this. 
In the **supervised method**, the attractors are known to the user, who provides one
initial condition for each attractor using the optional `attractors_ic`.
The algorithm then evolves these initial conditions, extracts their features, and uses them
as templates representing the attrators. Each trajectory is considered to belong to the
nearest template, which is found using a first-neighbor clustering algorithm.

If the attractors are not as well-known the **unsupervised method** should be used
instead, which means that the user does not provide the optional `attractors_ic` argument. 
Here, the vectors of features of each initial condition are mapped to an attractor by
analysing how the features are clustered in the feature space. Using the DBSCAN algorithm, 
we identify these clusters of features, and consider each cluster to represent an 
attractor. Features whose attractor is not identified are labeled as `-1`.
Otherwise, they are labeled starting from `1` in ascending order.

## Parallelization note
The trajectories in this method are integrated in parallel using `Threads`.
To enable this, simply start Julia with the number of threads you want to use.

[^Stender2021] : Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function basin_fractions_clustering(ds::DynamicalSystem, featurizer::Function,
    ics::Union{Dataset, Function}, attractors_ic::Union{Dataset, Nothing}=nothing; kwargs...)

    feature_array = extract_features_allics(ds, ics,  featurizer; kwargs...)

    if isnothing(attractors_ic) #unsupervised, no templates; 
        class_labels, class_errors = classify_solution(feature_array; kwargs...)
    else #supervised
        feature_templates = extract_features_allics(ds, attractors_ic, featurizer; kwargs...)
        class_labels, class_errors = classify_solution(feature_array, feature_templates; 
        kwargs...);
    end

    fs = basin_fractions(class_labels)
    if typeof(ics) <: Dataset return fs, class_labels end
    return fs #::Function ics
end


#----- INTEGRATION AND FEATURE EXTRACTION
"""
`extract_features_allics` receives the pre-generated initial conditions `ics` in a `Dataset`
and returns their extracted features in a matrix, with the j-th column containing the
j-th feature. To do this, it  calls the other `extract_features` method, made for just one array of ICs.
`ics` should contain each initial condition along its rows.
"""
function extract_features_allics(ds, ics::Dataset, featurizer::Function; show_progress=true,
    kwargs...)
    N = size(ics, 1) #number of actual ICs
    feature_array = Vector{Vector{Float64}}(undef, N)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    Threads.@threads for i = 1:N 
        ic = ics[i]
        feature_array[i] = extract_features(ds, ic, featurizer; kwargs...)
        show_progress && next!(progress)
    end
    return reduce(hcat, feature_array)
end

"""
`extract_features_allics` receives the sampler function to generate the initial conditions `ics`,
generates them and returns their extracted features in a matrix, with the j-th column containing the
j-th feature. To do this, it  calls the other `extract_features` method, made for just one array of ICs.
"""
function extract_features_allics(ds, ics::Function, featurizer::Function; N = 1000,
    show_progress=true, kwargs...)
    feature_array = Vector{Vector{Float64}}(undef, N)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    Threads.@threads for i = 1:N 
        ic = ics()
        feature_array[i] = extract_features(ds, ic, featurizer; kwargs...)
        show_progress && next!(progress)
    end
    return reduce(hcat, feature_array)
end


"""
`extract_features` receives an initial condition and returns its extracted features in a vector.
It integrates the initial condition, applies the `featurizer` function and returns
its output. The type of the returned vector depends on `featurizer`'s output.
"""
function extract_features(ds, u0, featurizer; T=100, Ttr=100, Δt=1, diffeq=NamedTuple(), kwargs...)
    u = trajectory(ds, T, u0; Ttr, Δt, diffeq) 
    t = Ttr:Δt:T+Ttr
    feature = featurizer(u, t)
    return feature
end
