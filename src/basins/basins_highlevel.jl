export draw_basin!, basins_of_attraction


"""
    basins_of_attraction(grid::Tuple, ds::DynamicalSystem; kwargs...) -> basins, attractors
Compute an estimate of the basins of attraction of a dynamical system `ds` on
a partitioning of the state space given by `grid`. The method has been
inspired by the 2D grid approach developed by Nusse & Yorke [^Yorke1997].
It works _without_ knowledge of where attractors are; it identifies them automatically.

The dynamical system can be:
* An actual `DiscreteDynamicalSystem` or `ContinuousDynamicalSystem`.
* A Poincaré map of `ContinuousDynamicalSystem`: [`poincaremap`](@ref).
* A stroboscopic map, i.e. a periodically forced `ContinuousDynamicalSystem` (see examples
  for this particular application).

`grid` is a tuple of ranges defining the grid of initial conditions, for example
`grid = (xg, yg)` where `xg = yg = range(-5, 5; length = 100)`. The grid is not necessarilly
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
* `Δt`: Approximate time step of the integrator, which is `1` for discrete systems.
  For continuous systems, it is recommended to use values such
  that one step will typically make the integrator move to a different cell of the
  state space partitioning. If this is not given (i.e., it is `nothing`, its default value),
  then an automatic estimation of `Δt` is done using the grid and the dynamical system.
  **Warning:** For ultra-fine grids, this can be even smaller than the internal step size
  of the integrator, if the solver is adaptive. In such a case, we advise to pick
  a non-adaptive solver and provide explicitly the argument `dt` in the `diffeq` keyword.
  TODO: Make function for Δt easy to use and expose it.
  If `nothing` (default), automatic estimation.... TODO
* `T` : Period of the stroboscopic map, in case of a continuous dynamical system with periodic
  time forcing. This argument is incompatible with `Δt`.
* `idxs = 1:length(grid)`: This vector selects the variables of the system that will define the
  subspace the dynamics will be projected into.
* `complete_state = zeros(D-Dg)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`, with `Dg` the dimension
  of the grid. It can be either a vector of length `D-Dg`, or a function `f(y)` that
  returns a vector of length `D-Dg` given the _projected_ initial condition on the grid `y`.
* `diffeq...`: Keyword arguments propagated to [`integrator`](@ref).
* `mx_chk_att = 2`: A parameter that sets the maximum checks of consecutives hits of an attractor
  before deciding the basin of the initial condition.
* `mx_chk_hit_bas = 10` : Maximum check of consecutive visits of the same basin of attraction.
  This number can be increased for higher accuracy.
* `mx_chk_fnd_att = 100` : Maximum check of unnumbered cell before considering we have an attractor.
  This number can be increased for higher accuracy.
* `mx_chk_loc_att = 100` : Maximum check of consecutive cells marked as an attractor before considering
  that we have all the available pieces of the attractor.
* `mx_chk_lost` : Maximum check of iterations outside the defined grid before we consider the orbit
  lost outside. This number can be increased for higher accuracy. It defaults to `20` if no
  attractors are given (see discussion on refining basins), and to `1000` if attractors are given.
* `horizon_limit = 1e6` : If the norm of the integrator state reaches this limit we consider that the
  orbit diverges.
* `show_progress = true` : By default a progress bar is shown using ProgressMeter.jl.
* `attractors, ε`: See discussion on refining basins below.

## Description
`basins` has the following organization:
* The basins are coded in sequential order from 1 up to the number of attractors.
* If the trajectory diverges or converges to an attractor outside the defined grid it is
  numbered `-1`

`attractors` has the following organization:
* The keys of the dictionary correspond to the number of the attractor.
* The value associated to a key is a [`Dataset`](@ref) with the *guessed* location of the
  attractor on the state space.

The method starts by picking the first available initial condition on the grid not yet
numbered. The dynamical system is then iterated until one of the following conditions happens:
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
that are close on the projected space but are far apart in another dimension. They could
be collapsed or confused into the same attractor. This is a drawback of this method.

## Refining basins of attraction
Sometimes one would like to be able to refine the found basins of attraction by recomputing
`basins_of_attraction` on a smaller, and more fine-grained, `grid`. If however this
new `grid` does not contain the attractors, `basins_of_attraction` would (by default)
attribute the value `-1` to all grid points. For these cases, an extra search clause can
be provided by setting the keywords `attractors, ε`. The `attractors` is a dictionary
mapping attractor IDs to `Dataset`s (i.e., the same as the return value of
`basins_of_attraction`). The algorithm checks at each step whether the system state is
`ε`-close (Euclidean norm) to any of the given attractors, and if so it attributes the stating grid point
to the basin of the close attractor. By default `ε` is equal to the mean grid spacing.

A word of advice while using this method: in order to work properly, `ε` should be
about the size of a grid cell that has been used to compute the given `attractors`. It is
recomended to keep the same step size (i.e., use the same integrator) since it may have an
influence in some cases. This algorithm is usually slower than the method with the
attractors on the grid.
"""
function basins_of_attraction(grid::Tuple, ds;
        Δt=nothing, T=nothing, idxs = 1:length(grid),
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid)),
        diffeq = NamedTuple(), kwargs...
        # `kwargs` tunes the basin finding algorithm, e.g. `mx_chk_att`.
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
    complete_and_reinit! = CompleteAndReinit(complete_state, idxs, length(get_state(integ)))
    get_projected_state = (integ) -> view(get_state(integ), idxs)

    fixed_solver = if haskey(kwargs, :diffeq) 
        haskey(kwargs[:diffeq], :dt) && haskey(kwargs[:diffeq], :adaptive)
    else
        false
    end

    if isnothing(Δt) && isnothing(T) && !fixed_solver
        Δt = automatic_Δt_basins(integ, grid, complete_and_reinit!)
        @show Δt
    end

    if !isnothing(T)
        iter_f! = (integ) -> step!(integ, T, true)
    elseif (integ isa PoincareMap) || fixed_solver
        iter_f! = step!
    else # generic case
        iter_f! = (integ) -> step!(integ, Δt) # we don't have to step _exactly_ `Δt` here
    end
    bsn_nfo = draw_basin!(
        grid, integ, iter_f!, complete_and_reinit!, get_projected_state; kwargs...
    )
    return bsn_nfo.basin, bsn_nfo.attractors
end

# Estimate Δt
using LinearAlgebra
function automatic_Δt_basins(integ, grid, complete_and_reinit!, N = 1000)
    if integ isa Union{PoincareMap, MinimalDiscreteIntegrator}
        return 1
    end
    steps = step.(grid)
    s = sqrt(sum(x^2 for x in steps)) # diagonal length of a cell
    indices = CartesianIndices(length.(grid))
    random_points = [generate_ic_on_grid(grid, ind) for ind in rand(indices, N)]
    dudt = 0.0
    udummy = copy(integ.u)
    for p in random_points
        complete_and_reinit!(integ, p)
        deriv = if integ.u isa SVector
            integ.f(integ.u, integ.p, 0.0)
        else
            integ.f(udummy, integ.u, integ.p, 0.0)
            udummy
        end
        dudt += norm(deriv)
    end
    return Δt = s*N/dudt
end
