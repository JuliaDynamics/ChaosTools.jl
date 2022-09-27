export basins_fractions_clustering, basins_fractions, cluster_datasets
using Statistics: mean
using Neighborhood
using ProgressMeter
include("clustering/cluster_config.jl")

#####################################################################################
# AttractorMapper API
#####################################################################################
struct AttractorsViaFeaturizing{DS<:GeneralizedDynamicalSystem, C<:ClusteringConfig, T, K, A,
    F} <: AttractorMapper
    ds::DS
    featurizer::F
    cluster_config::C
    Ttr::T
    Δt::T
    total::T
    diffeq::K
    attractors_ic::A
end

"""
    AttractorsViaFeaturizing(
        ds::DynamicalSystem, featurizer::Function,
        clusterconfig = ClusteringConfig(); kwargs...
    )

Initialize a `mapper` that maps initial conditions to attractors using the featurizing and
clustering method of [^Stender2021]. See [`AttractorMapper`](@ref) for how to use the
`mapper`.

`featurizer` is a function that takes as an input an integrated trajectory `A::Dataset` and
the corresponding time vector `t` and returns a `Vector{<:Real}` of features describing the
trajectory. See [`ClusteringConfig`](@ref) for configuring the clustering process.

## Keyword arguments
* `T=100, Ttr=100, Δt=1, diffeq=NamedTuple()`: Propagated to [`trajectory`](@ref).

## Description
The trajectory `X` of each initial condition is transformed into a vector of features. Each
feature is a number useful in _characterizing the attractor_ the initial condition ends up
at, and distinguishing it from other attrators. Example features are the mean or standard
deviation of one of the of the timeseries of the trajectory, the entropy of some of the
dimensions, the fractal dimension of `X`, or anything else you may fancy. The vectors of
features are then used to identify to which attractor each trajectory belongs (i.e. in which
basin of attractor each initial condition is in). The method thus relies on the user having
at least some basic idea about what attractors to expect in order to pick the right
features, in contrast to [`AttractorsViaRecurrences`](@ref).

Once the features are extracted, they are clustered using
[`ClusteringConfig`](@ref), so see that docstring for more details on the clustering.
Each cluster is considered one attractor.

If `templates` are provided to [`ClusteringConfig`](@ref), then a supervised version is used,
and the functionality is similar to [`AttractorsViaProximity`](@ref). Generally speaking, the
[`AttractorsViaProximity`](@ref) is superior. However, if the dynamical system has extremely
high-dimensionality, there may be reasons to use the supervised method of this featurizing
algorithm instead, as it projects the trajectories into a much lower dimensional
representation of features.

[^Stender2021]:
    Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
    stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function AttractorsViaFeaturizing(ds::GeneralizedDynamicalSystem, featurizer::Function,
    cluster_config::ClusteringConfig = ClusteringConfig(); T=100, Ttr=100, Δt=1,
    diffeq = NamedTuple(), attractors_ic::Union{AbstractDataset, Nothing}=nothing)
    if ds isa ContinuousDynamicalSystem
        T, Ttr, Δt = float.((T, Ttr, Δt))
    end
    return AttractorsViaFeaturizing(
        ds, featurizer, cluster_config, Ttr, Δt, T, diffeq, attractors_ic
    )
end

DynamicalSystemsBase.get_rule_for_print(m::AttractorsViaFeaturizing) =
get_rule_for_print(m.ds)

function Base.show(io::IO, mapper::AttractorsViaFeaturizing)
    ps = generic_mapper_print(io, mapper)
    println(io, rpad(" type: ", ps), nameof(typeof(mapper.ds)))
    println(io, rpad(" Ttr: ", ps), mapper.Ttr)
    println(io, rpad(" Δt: ", ps), mapper.Δt)
    println(io, rpad(" T: ", ps), mapper.total)
    return
end

ValidICS = Union{AbstractDataset, Function}

# We need to extend the general `basins_fractions`, because the clustering method
# cannot map individual initial conditions to attractors
function basins_fractions(mapper::AttractorsViaFeaturizing, ics::ValidICS;
        show_progress = true, N = 1000
    )
    feature_array = extract_features(mapper, ics; show_progress, N)
    cluster_labels  = cluster_features(feature_array, mapper.cluster_config)
    fs = basins_fractions(cluster_labels) # Vanilla fractions method with Array input
    if typeof(ics) <: AbstractDataset
        attractors = extract_attractors(mapper, cluster_labels, ics)
        return fs, cluster_labels, attractors
    else
        return fs
    end
end

function extract_features(mapper::AttractorsViaFeaturizing, ics::ValidICS;
    show_progress = true, N = 1000)

    N = (typeof(ics) <: Function)  ? N : size(ics, 1) # number of actual ICs

    feature_array = Vector{Vector{Float64}}(undef, N)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    # TODO: We can multi-thread this, but we need to make it so that integrators
    # are deepcopied (i.e., if mapper has a stroboscopic map)
    for i ∈ 1:N
        ic = _get_ic(ics,i)
        feature_array[i] = extract_features(mapper, ic)
        show_progress && ProgressMeter.next!(progress)
    end
    return feature_array
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