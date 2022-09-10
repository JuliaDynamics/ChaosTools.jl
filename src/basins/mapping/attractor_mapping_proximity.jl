"""
    AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict [, ε]; kwargs...)
Map initial conditions to attractors based on whether the trajectory reaches `ε`-distance
close to any of the user-provided `attractors`. They have to be in a form of a dictionary
mapping attractor labels to `Dataset`s containing the attractors.

The system gets stepped, and at each step the minimum distance to all
attractors is computed. If any of these distances is `< ε`, then the label of the nearest
attractor is returned.

If an `ε::Real` is _not_ provided by the user, a value is computed
automatically as half of the minimum distance between all attractors.
This operation can be expensive for large attractor datasets.
If `length(attractors) == 1`, then `ε` becomes 1/10 of the diagonal of the box containing
the attractor. If `length(attractors) == 1` and the attractor is a single point,
an error is thrown.

Because in this method the attractors are already known to the user,
the method can also be called _supervised_.

## Keywords
* `Ttr = 100`: Transient time to first evolve the system for before checking for proximity.
* `Δt = 1`: Integration step time (only valid for continuous systems).
* `horizon_limit = 1e3`: If the maximum distance of the trajectory from any of the given
  attractors exceeds this limit, it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `mx_chk_lost = 1000`: If the integrator has been stepped this many times without
  coming `ε`-near to any attractor,  it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `diffeq = NamedTuple()`: Keywords propagated to DifferentialEquations.jl
  (only valid for continuous systems).
"""
struct AttractorsViaProximity{I, AK, D, T, N, K} <: AttractorMapper
    integ::I
    attractors::Dict{AK, Dataset{D, T}}
    ε::Float64
    Δt::N
    Ttr::N
    mx_chk_lost::Int
    horizon_limit::Float64
    search_trees::K
    dist::Vector{Float64}
    idx::Vector{Int}
    maxdist::Float64
end
function AttractorsViaProximity(ds, attractors::Dict, ε = nothing;
        Δt=1, Ttr=100, mx_chk_lost=1000, horizon_limit=1e3, diffeq = NamedTuple(),
        verbose = false, kwargs...
    )
    @assert dimension(ds) == dimension(first(attractors)[2])
    search_trees = Dict(k => KDTree(att.data, Euclidean()) for (k, att) in attractors)
    integ = integrator(ds; diffeq)
    if isnothing(ε)
        ε = _deduce_ε_from_attractors(attractors, search_trees, verbose)
    else
        @assert ε isa Real
    end

    mapper = AttractorsViaProximity(
        integ, attractors,
        ε, Δt, eltype(Δt)(Ttr), mx_chk_lost, horizon_limit,
        search_trees, [Inf], [0], 0.0,
    )

    return mapper
end

function _deduce_ε_from_attractors(attractors, search_trees, verbose = false)
    if length(attractors) != 1
        verbose && @info("Computing minimum distance between attractors to deduce `ε`...")
        # Minimum distance between attractors
        # notice that we do not use `dataset_distance` because
        # we have more than two datasets and want the absolute minimum distance
        # between one of them.
        dist, idx = [Inf], [0]
        minε = Inf
        for (k, A) in attractors
            for (m, tree) in search_trees
                k == m && continue
                for p in A # iterate over all points of attractor
                    Neighborhood.NearestNeighbors.knn_point!(
                        tree, p, false, dist, idx, Neighborhood.alwaysfalse
                    )
                    dist[1] < minε && (minε = dist[1])
                end
            end
        end
        @info("Minimum distance between attractors computed: $(minε)")
        ε = minε/2
    else
        attractor = first(attractors)[2] # get the single attractor
        mini, maxi = minmaxima(attractor)
        ε = sqrt(sum(abs, maxi .- mini))/10
        if ε == 0
            throw(ArgumentError("""
            Computed `ε = 0` in automatic estimation, probably because there is
            a single attractor that also is a single point. Please provide `ε` manually.
            """))
        end
    end
    return ε
end


function (mapper::AttractorsViaProximity)(u0; show_progress = false)
    reinit!(mapper.integ, u0)
    maxdist = 0.0
    mapper.Ttr > 0 && step!(mapper.integ, mapper.Ttr)
    lost_count = 0
    while lost_count < mapper.mx_chk_lost
        step!(mapper.integ, mapper.Δt)
        lost_count += 1
        u = get_state(mapper.integ)
        for (k, tree) in mapper.search_trees # this is a `Dict`
            Neighborhood.NearestNeighbors.knn_point!(
                tree, u, false, mapper.dist, mapper.idx, Neighborhood.alwaysfalse
            )
            if mapper.dist[1] < mapper.ε
                return k
            elseif maxdist < mapper.dist[1]
                maxdist = mapper.dist[1]
                maxdist > mapper.horizon_limit && return -1
            end
        end
    end
    return -1
end

function Base.show(io::IO, mapper::AttractorsViaProximity)
    ps = generic_mapper_print(io, mapper)
    println(io, rpad(" type: ", ps), nameof(typeof(mapper.integ)))
    println(io, rpad(" ε: ", ps), mapper.ε)
    println(io, rpad(" Δt: ", ps), mapper.Δt)
    println(io, rpad(" Ttr: ", ps), mapper.Ttr)
    attstrings = split(sprint(show, MIME"text/plain"(), mapper.attractors), '\n')
    println(io, rpad(" attractors: ", ps), attstrings[1])
    for j in 2:length(attstrings)
        println(io, rpad(" ", ps), attstrings[j])
    end
    return
end

extract_attractors(mapper::AttractorsViaProximity, labels, ics) = mapper.attractors
