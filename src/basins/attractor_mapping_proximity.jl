"""
    AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; kwargs...)
Map initial conditions to attractors based on whether the trajectory reaches `ε`-distance
close to any of the user-provided `attractors`. They have to be in a form of a dictionary
mapping attractor labels to `Dataset`s containing the attractors.

The state of the system gets stepped, and at each step the minimum distance to all
attractors is computed. If any of these distances is `< ε`, then the label of the nearest
attractor is returned. 

Because in this method all possible attractors are already known to the user,
the method can also be called _supervised_.

## Keywords
* `Δt = 1`: Time covered by each step (only valid for continuous systems).
* `horizon_limit = 1e6`: If `norm(get_state(ds))` exceeds this number, it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `mx_chk_lost = 1000`: If the integrator has been stepped this many times without
  coming `ε`-near to any attractor,  it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `diffeq = NamedTuple()`: Keywords propagated to DifferentialEquations.jl
  (only valid for continuous systems).
"""
struct AttractorsViaProximity{I, F, D, T, K} <: AttractorMapper
    integ::I
    step!::F
    attractors::Dict{Int16, Dataset{D, T}}
    ε::Float64
    Δt::T
    mx_chk_lost::Int
    horizon_limit::T
    search_trees::K
    dist::Vector{Float64}
    idx::Vector{Int}
end
function AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; 
        ε=1e-3, Δt=1, mx_chk_lost=1000, horizon_limit=1e6, diffeq = NamedTuple()
    )
    @assert dimension(ds) == dimension(first(attractors))
    search_trees = Dict(k => KDTree(att.data, Euclidean()) for (k, att) in attractors)

    # TODO: After creating StroboscopicMap, the API will be simpler here...?

    # TODO: For Poincare/Stroboscopic maps, `integrator` should return the objects
    # themselves
    fixed_solver = haskey(diffeq, :dt)
    integ = integrator(ds; diffeq)
    if !(isdiscretesystem(ds) || fixed_solver)
        iter_f! = (integ) -> step!(integ, Δt)
    else
        iter_f! = (integ) -> step!(integ)
    end

    return AttractorsViaProximity(
        integ, iter_f!, attractors, 
        ε, Δt, mx_chk_lost, horizon_limit, 
        search_trees, [Inf], [0],
    )
end

function (mapper::AttractorsViaProximity)(u0)
    reinit!(mapper.integ, u0)
    lost_count = 0
    while lost_count < mapper.mx_ch_lost
        mapper.step!(mapper.integ)
        lost_count += 1
        u = get_state(integ)
        if lost_count > mx_chk_lost || norm(u) > horizon_limit
            return -1
        end
        for (k, t) in mapper.search_trees # this is a `Dict`
            Neighborhood.NearestNeighbors.knn_point!(
                t, u, false, mapper.dist, mapper.idx, Neighborhood.alwaysfalse
            )
            if mapper.dist[1] < ε
                return k
            end
        end
    end
end
