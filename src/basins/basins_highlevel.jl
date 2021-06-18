export draw_basin!, basins_of_attraction


"""
    basins_of_attraction(grid::Tuple, ds::DynamicalSystem; kwargs...) -> basins, attractors
Compute an estimate of the basins of attraction of a dynamical system `ds` on
a partitioning of the state space given by `grid`. The method has been
inspired by the 2D grid approach devellopped by Nusse & Yorke [^Yorke1997].

The dynamical system can be:
* An actual `DiscreteDynamicalSystem` or `ContinuousDynamicalSystem`.
* A Poincaré map of `ContinuousDynamicalSystem`.
* A stroboscopic map, i.e. a periodically forced `ContinuousDynamicalSystem` (see examples
  for this particular application).

`grid` is a tuple of ranges defining the grid of initial conditions, for example
`grid=(xg,yg)` where `xg` and `yg` are one dimensional ranges. The grid is not necessarilly
of the same dimension as the dynamical system, the attractors can be found in lower dimensional
projections.
`ds` is a `DynamicalSystem`. This includes the output of [`poincaremap`](@ref).
See documentation online for examples for all cases!
The output `basins` is an array on the grid (`xg, yg`), see below for details.
The output `attractors` is a dictionary whose keys correspond to the attractor number and
the values contains the points of the attractors found on the map. Notice that for some
attractors this list may be incomplete.

See also [`match_attractors!`](@ref), [`basin_fractions`](@ref), [`tipping_probabilities`](@ref).

[^Yorke1997]: H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations Ch. 7, Springer, New York, 1997

## Keyword Arguments
* `dt = 1`: Approximate time step of the integrator. It is recommended to use values such
  that one step will typically make the integrator move to a different cell of the
  state space partitioning.
* `T=0` : Period of the stroboscopic map, in case of a continuous dynamical system with periodic
   time forcing. This argument is incompatible with `dt`. 
* `idxs = 1:2`: This vector selects the two variables of the system that will define the
  "plane" the dynamics will be projected into.
* `complete_state = zeros(D-Dg)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`, beeing `Dg` the dimension
  of the grid. It can be either a vector of length `D-Dg`, or a function `f(x, y)` that
  returns a vector of length `D-Dg`.
* `diffeq...`: Keyword arguments propagated to [`integrator`](@ref).
* `mx_chk_att = 2`: A parameter that sets the maximum checks of consecutives hits of an attractor
   before deciding the basin of the initial condition.
* `mx_chk_hit_bas = 10` : Maximum check of consecutive visits of the same basin of attraction.
  This number can be increased for higher accuracy.
* `mx_chk_fnd_att = 60` : Maximum check of unnumbered cell before considering we have an attractor.
   This number can be increased for higher accuracy.
* `mx_chk_lost = 2000` : Maximum check of iterations outside the defined grid before we consider the orbit
   lost outside. This number can be increased for higher accuracy.
* `horizon_limit = 10^10` : If the norm of the integrator state reaches this limit we consider that the
   orbit diverges.

## Description
This method identifies the attractors and their basins of attraction on the grid without
prior knowledge about the system. At the end of a successfull computation the function
returns an Array coding the basins of attraction and a dictionary with all attractors found.

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
1. The trajectory diverges or hits an attractor outside the defined grid: the initial
   condition is set to -1
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

This function can be used to make attractor basins in any dimension. For example:
```julia
xg = yg = zg = 0:0.01:1 # the range defining the z part of the grid
b, a = basins_of_attraction((xg, yg, zg), ds; complete_state = [0.0])
```
"""
function basins_of_attraction(grid::Tuple, ds::DynamicalSystem;
        dt=1, T=0, idxs = SVector(1, 2), # TODO: `idxs` must have same length as `grid`
        complete_state=zeros(dimension(ds)-2), diffeq = NamedTuple(),
        kwargs... # `kwargs` tunes the basin finding algorithm, e.g. `mx_chk_att`.
                  # these keywords are actually expanded in `draw_basin!`
    )
    integ = integrator(ds; diffeq...)
    idxs = SVector(idxs...)
    return basins_of_attraction(grid, integ, dt, T, idxs, complete_state; kwargs...)
end

function basins_of_attraction(grid::Tuple, pmap::PoincareMap; kwargs...)
    reinit_f! = (pmap,y) -> _init_map(pmap, y, pmap.i)
    get_u = (pmap) -> pmap.integ.u[pmap.i]
    bsn_nfo = draw_basin!(grid, pmap, step!, reinit_f!, get_u; kwargs...)
    return bsn_nfo.basin, bsn_nfo.attractors
end

function _init_map(pmap::PoincareMap, y, idxs)
    u = zeros(1,length(pmap.integ.u))
    u[idxs] = y
    reinit!(pmap, u)
end

function basins_of_attraction(grid, integ, dt, T, idxs::SVector, complete_state; kwargs...)
    if T>0
        iter_f! = (integ) -> step!(integ, T, true)
    else
        iter_f! = (integ) -> step!(integ, dt)# we don't have to step _exactly_ `dt` here
    end
    D = length(integ.u)
    remidxs = setdiff(1:D, idxs)

    if complete_state isa AbstractVector
        if D == length(idxs)
            complete_state=[]
        elseif length(complete_state) ≠ D-length(idxs)
             error("Vector `complete_state` must have length D-Dg!")
        end
        u0 = copy(complete_state)
        reinit_f! = (integ, y) -> reinit_integ_idxs!(integ, y, idxs, u0, remidxs)
    elseif complete_state isa Function
        y = ones(1,length(grid))
        u = complete_state(y...)
        !(typeof(u) <: StaticArray) && error("The function `complete_state` must return a Static vector")
        reinit_f! = (integ, z) -> begin
            u0 = complete_state(z...)
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
