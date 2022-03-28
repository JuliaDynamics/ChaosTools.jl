# TODO:
# This entire file is DEPRECATED.

"""
    basins_of_attraction(grid::Tuple, ds::GeneralizedDynamicalSystem; kwargs...)
Compute an estimate of the basins of attraction of a dynamical system `ds` on
a partitioning of the state space given by `grid`.
This function implements the method developed by Datseris & Wagemakers [^Datseris2022].

Works for any case encapsulated by [`GeneralizedDynamicalSystem`](@ref).


It works _without_ knowledge of where attractors are; it identifies them automatically.

The dynamical system an actual `DynamicalSystem`, or any
of the [Available integrators](@ref), such as e.g., a [`stroboscopicmap`](@ref).

`grid` is a tuple of ranges defining the grid of initial conditions, for example
`grid = (xg, yg)` where `xg = yg = range(-5, 5; length = 100)`.
The grid has to be the same dimensionality as the state space, use a
[`projected_integrator`](@ref) if you want to search for attractors in a lower
dimensional space.

The function returns `basins, attractors`. `basins` is an integer-valued array on the
`grid`, with its entries labelling
which basin of attraction the given grid point belongs to.
The output `attractors` is a dictionary whose keys correspond to the attractor number and
the values contains the points of the attractors found.
Notice that for some attractors this list may be incomplete.

See also [`match_attractors!`](@ref), [`basin_fractions`](@ref), [`tipping_probabilities`](@ref).

[^Datseris2022]:
    G. Datseris and A. Wagemakers, *Effortless estimation of basins of attraction*,
    [Chaos 32, 023104 (2022)](https://doi.org/10.1063/5.0076568)

## Keyword Arguments
* `Δt`: Approximate time step of the integrator, which is `1` for discrete systems.
  For continuous systems, an automatic value is calculated using [`automatic_Δt_basins`](@ref).
  See that function for more info.
* `T` : Period of the stroboscopic map, in case of a continuous dynamical system with periodic
  time forcing. This argument is incompatible with `Δt`.
* `idxs = 1:length(grid)`: This vector selects the variables of the system that will define the
  subspace the dynamics will be projected into.
* `complete_state = zeros(D-Dg)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`, with `Dg` the dimension
  of the grid. It can be either a vector of length `D-Dg`, or a function `f(y)` that
  returns a vector of length `D-Dg` given the _projected_ initial condition on the grid `y`.
* `diffeq = NamedTuple()`: Keyword arguments propagated to [`integrator`](@ref). Only
  useful for continuous systems. It is **strongly recommended** to choose high accuracy
  solvers for this application, e.g. `diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)`.
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
        Δt=nothing, diffeq = NamedTuple(), attractors = nothing,
        # T, idxs, compelte_state are DEPRECATED!
        T=nothing, idxs = nothing, complete_state = nothing,
        # `kwargs` tunes the basin finding algorithm, e.g. `mx_chk_att`.
        # these keywords are actually expanded in `_identify_basin_of_cell!`
        kwargs...
    )

    @warn("""
    The function `basins_of_attraction(grid::Tuple, ds::DynamicalSystem; ...)` is
    deprecated in favor of the simpler and more generic
    `basins_of_attraction(mapper::AttractorMapper, grid::Tuple`) which works for
    any instance of `AttractorMapper`. Please use that method in the future.
    """)

    if !isnothing(T)
        @warn("Using `T` is deprecated. Initialize a `stroboscopicmap` and pass it.")
        integ = stroboscopicmap(ds, T)
    elseif ds isa PoincareMap
        integ = ds
    elseif length(grid) ≠ dimension(ds) && isnothing(idxs)
        @warn("Giving a `grid` with dimension lower than `ds` is deprecated. "*
        "Initialize a `projected_integrator` instead.")
        idxs = 1:length(grid)
        c = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
        integ = projected_integrator(ds, idxs, c; diffeq)
    elseif !isnothing(idxs)
        @warn("Using `idxs` is deprecated. Initialize a `projeted_integrator` instead.")
        @assert length(idxs) == length(grid)
        idxs = 1:length(grid)
        if isnothing(complete_state)
            c = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
        else
            c = complete_state
        end
        integ = projected_integrator(ds, idxs, c; diffeq)
    else
        integ = ds
    end

    if !isnothing(attractors) # proximity version
        # initialize Proximity and loop.
        mapper = AttractorsViaProximity(integ, attractors::Dict, ε;
        Δt=isnothing(Δt) ? 1 : Δt, Ttr, mx_chk_lost, horizon_limit, diffeq,)
        return estimate_basins_proximity!(mapper, grid; kwargs...)
    else # (original) recurrences version
        bsn_nfo, integ = basininfo_and_integ(integ, grid, Δt, diffeq)
        bsn_nfo = estimate_basins_recurrences!(grid, bsn_nfo, integ; kwargs...)
        return bsn_nfo.basins, bsn_nfo.attractors
    end
end


import ProgressMeter
using Statistics: mean

function estimate_basins_proximity!(mapper, grid; show_progress = true)
    basins = zeros(Int16, map(length, grid))
    progress = ProgressMeter.Progress(
        length(basins); desc = "Basins of attraction: ", dt = 1.0
    )
    for (k,ind) in enumerate(CartesianIndices(bsn_nfo))
        show_progress && ProgressMeter.update!(progress, k)
        y0 = generate_ic_on_grid(grid, ind)
        basins[ind] = mapper(y0)
    end
    return basins, attractors
end


"""
This is the low level function that computes the full basins of attraction,
given the already initialized `BasinsInfo` object and the integrator.
It simply loops over the `get_label_ic!` function, that maps initial conditions
to attractors.
"""
function estimate_basins_recurrences!(
        grid::Tuple,
        bsn_nfo::BasinsInfo, integ;
        show_progress = true, kwargs...,
    )
    I = CartesianIndices(bsn_nfo.basins)
    progress = ProgressMeter.Progress(
        length(bsn_nfo.basins); desc = "Basins of attraction: ", dt = 1.0
    )

    for (k,ind) in enumerate(I)
        if bsn_nfo.basins[ind] == 0
            show_progress && ProgressMeter.update!(progress, k)
            y0 = generate_ic_on_grid(grid, ind)
            bsn_nfo.basins[ind] =
            get_label_ic!(bsn_nfo, integ, y0; show_progress, kwargs...)
        end
    end

    # remove attractors and rescale from 1 to max number of attractors
    ind = iseven.(bsn_nfo.basins)
    bsn_nfo.basins[ind] .+= 1
    bsn_nfo.basins .= (bsn_nfo.basins .- 1) .÷ 2

    return bsn_nfo
end
