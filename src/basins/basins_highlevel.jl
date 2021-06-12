export draw_basin!, basins_2D, basins_general

"""
    basins_2D(xg, yg, integ; kwargs...) → basins, attractors
Compute an estimate of the basins of attraction of a "two dimensional system"
of the plane onto itself according to the method of Nusse & Yorke[^Yorke1997].
The dynamical system can be:
* An actual 2D `DiscreteDynamicalSystem` or `ContinuousDynamicalSystem`.
* 2D poincaré map of a 3D `ContinuousDynamicalSystem`.
* A 2D stroboscopic map, i.e. a periodically forced 2D `ContinuousDynamicalSystem`.

For a higher-dimensional dynamical systems, use [`basins_general`](@ref).

`integ` is an istance of an integrator, not a `DynamicalSystem`. This includes
the output of [`poincaremap`](@ref). See documentation online for examples for all cases!
`xg`, `yg` are 1-dimensional ranges that define the grid of the initial conditions
to test.
The output `basins` is a matrix on the grid (`xg, yg`), see below for details.
The output `attractors` is a dictionary whose keys correspond to the attractor number and
the values contains the points of the attractors found on the map. Notice that for some
attractors this list may be incomplete.

See also [`match_attractors!`](@ref), [`basin_fractions`](@ref), [`tipping_probabilities`](@ref).

[^Yorke1997]: H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations Ch. 7, Springer, New York, 1997

## Keyword Arguments
* `T` : Period of the stroboscopic map, in case `integ` is an integrator of a 2D continuous
  dynamical system with periodic time forcing.
* `mc_att = 10`: A parameter that sets the maximum checks of consecutives hits of an attractor
  before deciding the basin of the initial condition.
* `mc_bas = 10` : Maximum check of consecutive visits of the same basin of attraction.
  This number can be increased for higher accuracy.
* `mc_unmb = 60` : Maximum check of unnumbered cell before considering we have an attractor.
  This number can be increased for higher accuracy.

## Description
This method identifies the attractors and their basins of attraction on the grid without
prior knowledge about the system. At the end of a successfull computation the function
returns a matrix coding the basins of attraction and a dictionary with all attractors found.

`basins` has the following organization:
* The basins are coded in sequential order from 1 to the number of attractors `Na` .
* If the trajectory diverges or converges to an attractor outside the defined grid it is
  numbered `-1`

`attractors` has the following organization:
* The keys of the dictionary correspond to the number of the attractor.
* The value associated to a key is a [`Dataset`](@ref) with the *guessed* location of the
  attractor on the state space.

The method starts by picking the first available initial condition on the plane not yet
numbered. The dynamical system is then iterated until one of the following conditions
happens:
1. The trajectory hits a known attractor already numbered `mc_att` consecutive times: the initial condition is
   numbered with the corresponding number.
1. The trajectory diverges or hits an attractor outside the defined grid: the initial
   condition is set to -1
1. The trajectory hits a known basin `mc_bas` times in a row: the initial condition belongs to
   that basin and is numbered accordingly.
1. The trajectory hits `mc_unmb` times in a row an unnumbered cell: it is considered an attractor
   and is labelled with a new number.

Regarding performace, this method is at worst as fast as tracking the attractors.
In most cases there is a signicative improvement in speed.
"""
function basins_2D(xg, yg, pmap::PoincareMap; kwargs...)
    reinit_f! = (pmap,y) -> _init_map(pmap, y, pmap.i)
    get_u = (pmap) -> pmap.integ.u[pmap.i]
    bsn_nfo = draw_basin!((xg, yg), pmap, step!, reinit_f!, get_u; kwargs...)
    return bsn_nfo.basin, bsn_nfo.attractors
end

function _init_map(pmap::PoincareMap, y, idxs)
    u = zeros(1,length(pmap.integ.u))
    u[idxs] = y
    # all other coordinates are zero
    reinit!(pmap, u)
end

function basins_2D(xg, yg, integ; T=nothing, kwargs...)
    if T isa Real
        iter_f! = (integ) -> step!(integ, abs(T), true)
    elseif isnothing(T)
        iter_f! = (integ) -> step!(integ)
    end
    reinit_f! = (integ,y) -> reinit!(integ, y)
    get_u = (integ) -> integ.u

    bsn_nfo = draw_basin!((xg, yg), integ, iter_f!, reinit_f!, get_u; kwargs...)
    return bsn_nfo.basin, bsn_nfo.attractors
end


"""
    basins_general(grid::Tuple, ds::DynamicalSystem; kwargs...) -> basin, attractors
Compute an estimate of the basins of attraction of a dynamical system `ds` on
a partitioning of the state space given by `grid`.
`grid` in tuple of ranges defining the grid of initial conditions
, for example `grid=[xg,yg]` where `xg` and `yg` are one dimensional ranges. Refer to
[`basins_2D`](@ref) for more details regarding the algorithm.

# TODO: All of this needs to be re-written, as we no longer project on 2D.
Notice that in the case we have to project the dynamics on a lower dimensional space,
there are edge cases where the system may have two attractors
that are close on the defined space but are far apart in another dimension. They could
be collapsed or confused into the same attractor. This is a drawback of this method.

This function can be used to make attractor basins in any dimension. For example:
```julia
xg = yg = zg = 0:0.01:1 # the range defining the z part of the grid
b, a = basins_general([xg, yg, zg], ds; complete_state = [0.0])
```

## Keyword Arguments
* `dt = 1`: Approximate time step of the integrator. It is recommended to use values such
  that one step will typically make the integrator move to a different cell of the 
  state space partitioning.
* `idxs = 1:2`: This vector selects the two variables of the system that will define the
  "plane" the dynamics will be projected into.
* `complete_state = zeros(D-Nu)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u` of length `Nu`. It can be
  either a vector of length `D-Nu`, or a function `f(x, y)` that returns a vector of
  length `D-Nu`.
* `mc_att, mc_bas, mc_unmb`: As in [`basins_2D`](@ref).
* `diffeq...`: Keyword arguments propagated to [`integrator`](@ref).
"""
function basins_general(grid::Tuple, ds::DynamicalSystem;
        dt=1, idxs = SVector(1, 2), # TODO: `idxs` must have same length as `grid`
        complete_state=zeros(dimension(ds)-2), diffeq = NamedTuple(),
        kwargs... # `kwargs` tunes the basin finding algorithm, e.g. `mc_att`.
                  # these keywords are actually expanded in `draw_basin!`
    )
    integ = integrator(ds; diffeq...)
    idxs = SVector(idxs...)
    return basins_general(grid, integ, dt, idxs, complete_state; kwargs...)
end

function basins_general(grid, integ, dt, idxs::SVector, complete_state; kwargs...)
    iter_f! = (integ) -> step!(integ, dt) # we don't have to step _exactly_ `dt` here
    D = length(integ.u)
    remidxs = setdiff(1:D, idxs)
    # TODO: We should really check here that our functions that complete the state
    # return static vectors
    if complete_state isa AbstractVector
        length(complete_state) ≠ D-length(idxs) && error("Vector `complete_state` must have length D-2!")
        u0 = copy(complete_state)
        reinit_f! = (integ, y) -> reinit_integ_idxs!(integ, y, idxs, u0, remidxs)
    elseif complete_state isa Function
        reinit_f! = (integ, z) -> begin
            x, y = z
            u0 = complete_state(x, y)
            return reinit_integ_idxs!(integ, z, idxs, u0, remidxs)
        end
    else
        error("Incorrect type for `complete_state`")
    end
    get_u = (integ) -> integ.u[idxs]
    bsn_nfo = draw_basin!(grid, integ, iter_f!, reinit_f!, get_u; kwargs...)
    return bsn_nfo.basin, bsn_nfo.attractors
end

"""
    reinit_integ_idxs!(integ, y, idxs, u, remidxs)
`reinit!` given integrator by setting its `idxs` entries of the state as
`y`, and the `remidxs` ones as `u`.
"""
function reinit_integ_idxs!(integ, y, idxs, u, remidxs)
    D = length(integ.u)
    s = zeros(D)
    s[idxs] .= y
    s[remidxs] .= u
    reinit!(integ, s)
end
