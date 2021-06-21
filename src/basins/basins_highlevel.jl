export draw_basin!, basins_of_attraction


"""
    basins_of_attraction(grid::Tuple, ds::DynamicalSystem; kwargs...) -> basins, attractors
Compute an estimate of the basins of attraction of a dynamical system `ds` on
a partitioning of the state space given by `grid`. The method has been
inspired by the 2D grid approach devellopped by Nusse & Yorke [^Yorke1997].
It works _without_ knowledge of where attractors are; it identifies them automatically.

The dynamical system can be:
* An actual `DiscreteDynamicalSystem` or `ContinuousDynamicalSystem`.
* A Poincaré map of `ContinuousDynamicalSystem`: [`poincaremap`](@ref).
* A stroboscopic map, i.e. a periodically forced `ContinuousDynamicalSystem` (see examples
  for this particular application).

`grid` is a tuple of ranges defining the grid of initial conditions, for example
`grid=(xg,yg)` where `xg` and `yg` are one dimensional ranges. The grid is not necessarilly
of the same dimension as the state space, attractors can be found in lower dimensional
projections.

The output `basins` is an integer-valued array on the `grid`, with its entries labelling
which basin of attraction the given grid point belongs to.
The output `attractors` is a dictionary whose keys correspond to the attractor number and
the values contains the points of the attractors found.
Notice that for some attractors this list may be incomplete.

See also [`match_attractors!`](@ref), [`basin_fractions`](@ref), [`tipping_probabilities`](@ref).

[^Yorke1997]: H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations Ch. 7, Springer, New York, 1997

## Keyword Arguments
* `Δt = 1`: Approximate time step of the integrator. It is recommended to use values such
  that one step will typically make the integrator move to a different cell of the
  state space partitioning.
* `T=0` : Period of the stroboscopic map, in case of a continuous dynamical system with periodic
  time forcing. This argument is incompatible with `Δt`. 
* `idxs = 1:length(grid)`: This vector selects the variables of the system that will define the
  subspace the dynamics will be projected into.
* `complete_state = zeros(D-Dg)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`, beeing `Dg` the dimension
  of the grid. It can be either a vector of length `D-Dg`, or a function `f(y)` that
  returns a vector of length `D-Dg` given the _projected_ initial condition on the grid `y`.
* `diffeq...`: Keyword arguments propagated to [`integrator`](@ref).
* `mx_chk_att = 2`: A parameter that sets the maximum checks of consecutives hits of an attractor
  before deciding the basin of the initial condition.
* `mx_chk_hit_bas = 10` : Maximum check of consecutive visits of the same basin of attraction.
  This number can be increased for higher accuracy.
* `mx_chk_fnd_att = 60` : Maximum check of unnumbered cell before considering we have an attractor.
  This number can be increased for higher accuracy.
* `mx_chk_lost = 1000` : Maximum check of iterations outside the defined grid before we consider the orbit
  lost outside. This number can be increased for higher accuracy.
* `horizon_limit = 1e6` : If the norm of the integrator state reaches this limit we consider that the
  orbit diverges.

## Description
`basins` has the following organization:
* The basins are coded in sequential order from 1 up to the number of attractors.
* If the trajectory diverges or converges to an attractor outside the defined grid it is
  numbered `-1`

`attractors` has the following organization:
* The keys of the dictionary correspond to the number of the attractor.
* The value associated to a key is a [`Dataset`](@ref) with the *guessed* location of the
  attractor on the state space.

The method starts by picking the first available initial condition on the plane not yet
numbered. The dynamical system is then iterated until one of the following conditions
happens:
1. The trajectory hits a known attractor already numbered `mx_chk_att` consecutive times: the
   initial condition is numbered with the corresponding number.
1. The trajectory spends `mx_chk_lost` steps outside the defiend grid: the initial
   condition is set to -1.
1. The trajectory hits a known basin `mx_chk_hit_bas` times in a row: the initial condition
   belongs to that basin and is numbered accordingly.
1. The trajectory hits `mx_chk_fnd_att` times in a row an unnumbered cell: it is considered
   an attractor and is labelled with a new number.

Regarding performace, this method is at worst as fast as tracking the attractors.
In most cases there is a signicative improvement in speed.

Notice that in the case we have to project the dynamics on a lower dimensional space,
there are edge cases where the system may have two attractors
that are close on the defined space but are far apart in another dimension. They could
be collapsed or confused into the same attractor. This is a drawback of this method.
"""
function basins_of_attraction(grid::Tuple, ds;
        Δt=1, T=0, idxs = 1:length(grid),
        complete_state=zeros(length(get_state(ds))-length(grid)), diffeq = NamedTuple(),
        kwargs... # `kwargs` tunes the basin finding algorithm, e.g. `mx_chk_att`.
                  # these keywords are actually expanded in `_identify_basin_of_cell!`
    )
    @assert length(idxs) == length(grid)
    integ = ds isa PoincareMap ? ds : integrator(ds; diffeq...)
    idxs = SVector(idxs...)
    return basins_of_attraction(grid, integ, Δt, T, idxs, complete_state; kwargs...)
end

function basins_of_attraction(grid, integ, Δt, T, idxs::SVector, complete_state; kwargs...)
    D = length(get_state(integ))
    if complete_state isa AbstractVector && (length(complete_state) ≠ D-length(idxs))
        error("Vector `complete_state` must have length D-Dg!")
    end
    if T>0
        iter_f! = (integ) -> step!(integ, T, true)
    elseif integ isa PoincareMap
        iter_f! = step!
    else # generic case
        iter_f! = (integ) -> step!(integ, Δt) # we don't have to step _exactly_ `Δt` here
    end
    complete_and_reinit! = CompleteAndReinit(complete_state, idxs, length(get_state(integ)))
    get_projected_state = (integ) -> view(get_state(integ), idxs)
    bsn_nfo = draw_basin!(grid, integ, iter_f!, complete_and_reinit!, get_projected_state; kwargs...)
    return bsn_nfo.basin, bsn_nfo.attractors
end

"""
    CompleteAndReinit(complete_state, idxs, D)
Helper struct that completes a state and reinitializes the integrator once called
as a function with arguments `f(integ, y)` with `integ` the initialized dynamical
system integrator and `y` the projected initial condition on the grid.
"""
struct CompleteAndReinit{C, Y, R}
    complete_state::C
    u::Vector{Float64} # dummy variable for a state in full state space
    idxs::SVector{Y, Int}
    remidxs::R
end
function CompleteAndReinit(complete_state, idxs, D::Int)
    remidxs = setdiff(1:D, idxs)
    remidxs = isempty(remidxs) ? nothing : SVector(remidxs...)
    u = zeros(D)
    return CompleteAndReinit(complete_state, u, idxs, remidxs)
end
function (c::CompleteAndReinit{<: AbstractVector})(integ, y)
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state
    end
    reinit!(integ, c.u)
end
function (c::CompleteAndReinit)(integ, y) # case where `complete_state` is a function
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state(y)
    end
    reinit!(integ, c.u)
end
