export draw_basin!, basins_map2D, basins_general, match_attractors!

mutable struct BasinInfo{F,T,Q}
    basin::Matrix{Int16}
    xg::F
    yg::F
    iter_f!::Function
    reinit_f!::Function
    get_u::Function
    current_color::Int
    next_avail_color::Int
    consecutive_match::Int
    consecutive_other_basins::Int
    prevConsecutives::Int
    prev_attr::Int
    prev_bas::Int
    prev_step::Int
    step::Int
    attractors::Dict{Int16, Dataset{2, T}}
    visited::Q
end

function Base.show(io::IO, bsn::BasinInfo)
    println(io, "Basin of attraction structure")
    println(io,  rpad(" size : ", 14),    size(bsn.basin))
    println(io,  rpad(" Number of attractors found: ", 14),   Int((bsn.current_color-2)/2)  )
end

"""
    basins_map2D(xg, yg, integ; kwargs...) → basins, attractors
Compute an estimate of the basins of attraction on a two-dimensional plane using a map
of the plane onto itself according to the method of Nusse & Yorke[^Yorke1997].
The dynamical system should be a discrete two dimensional system such as:
* Discrete 2D map.
* 2D poincaré map.
* A 2D stroboscopic map.
For a higher-dimensional dynamical system, use [`basins_general`](@ref).

`integ` is an istance of an integrator, not a `DynamicalSystem`. This includes
the output of [`poincaremap`](@ref). See documentation online for examples for all cases!
`xg`, `yg` are 1-dimensional ranges that define the grid of the initial conditions
to test.
The output `basins` is a matrix on the grid (`xg, yg`), see below for details.
The output `attractors` is a dictionary whose keys correspond to the attractor number and
the values contains the points of the attractors found on the map. Notice that for some
attractors this list may be incomplete.

See also [`basin_fractions`](@ref), [`tipping_probabilities`](@ref).

[^Yorke1997]: H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations Ch. 7, Springer, New York, 1997

## Keyword Arguments
* `T` : Period of the stroboscopic map, in case `integ` is an integrator of a 2D continuous
  dynamical system with periodic time forcing.
* `mc_att = 10`: A parameter that sets the maximum checks of consecutives hits of an attractor
  before deciding the basin of the initial condition.
* `mc_bas = 10` : Maximum check of consecutive visits of the same basin of attraction. This number
  can be increased for higher accuracy.
* `mc_unmb = 60` : Maximum check of unnumbered cell before considering we have an attractor. This number
  can be increased for higher accuracy.

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
  attractor on the plane.

The method starts by picking the first available initial condition on the plane not yet
numbered. The dynamical system is then iterated until one of the following conditions
happens:
1. The trajectory hits a known attractor already numbered: the initial condition is
   numbered with corresponding odd number.
1. The trajectory diverges or hits an attractor outside the defined grid: the initial
   condition is set to -1
1. The trajectory hits a known basin `mc_bas` times in a row: the initial condition belongs to
   that basin and is numbered accordingly.
1. The trajectory hits `mc_unmb` times in a row an unnumbered cell: it is considered an attractor
   and is labelled with a even number.

Regarding performace, this method is at worst as fast as tracking the attractors.
In most cases there is a signicative improvement in speed.
"""
function basins_map2D(xg, yg, pmap::PoincareMap; mc_att = 3, mc_bas = 10, mc_unmb = 60)
    reinit_f! = (pmap,y) -> _init_map(pmap, y, pmap.i)
    get_u = (pmap) -> pmap.integ.u[pmap.i]
    bsn_nfo = draw_basin!(xg, yg, pmap, step!, reinit_f!, get_u, mc_att, mc_bas, mc_unmb)
    return bsn_nfo.basin, bsn_nfo.attractors
end

function _init_map(pmap::PoincareMap, y, idxs)
    u = zeros(1,length(pmap.integ.u))
    u[idxs] = y
    # all other coordinates are zero
    reinit!(pmap, u)
end

function basins_map2D(xg, yg, integ; T=nothing, mc_att = 2, mc_bas = 10, mc_unmb = 60)
    if T isa Real
        iter_f! = (integ) -> step!(integ, abs(T), true)
    elseif isnothing(T)
        iter_f! = (integ) -> step!(integ)
    end
    reinit_f! =  (integ,y) -> reinit!(integ, y)
    get_u = (integ) -> integ.u

    bsn_nfo = draw_basin!(xg, yg, integ, iter_f!, reinit_f!, get_u, mc_att, mc_bas, mc_unmb)
    return bsn_nfo.basin, bsn_nfo.attractors
end


"""
    basins_general(xg, yg, ds::DynamicalSystem; kwargs...) -> basin, attractors
Compute an estimate of the basins of attraction of a higher-dimensional dynamical system `ds`
on a projection of the system dynamics on a two-dimensional plane.

Like [`basins_map2D`](@ref), `xg, yg` are ranges defining the grid of initial conditions
on the plane.

Refer to [`basins_map2D`](@ref) for detailed information on the
computation and the structure of `basin` and `attractors`.

## Keyword Arguments
* `dt = 1`: Approxiamte time step of the integrator. It is recommended to use values ≥ 1.
* `idxs = 1:2`: This vector selects the two variables of the system that will define the
  "plane" the dynamics will be projected into.
* `complete_state = zeros(D-2)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each planar initial condition `x, y`. It can be
  either a vector of length `D-2`, or a function `f(x, y)` that returns a vector of
  length `D-2`.
* `mc_att, mc_bas, mc_unmb`: As in [`basins_map2D`](@ref).
* `diffeq...`: Keyword arguments propagated to [`integrator`](@ref).
"""
function basins_general(xg, yg, ds::DynamicalSystem;
        dt=1, idxs = SVector(1, 2), mc_att = 10, mc_bas = 10, mc_unmb = 60, complete_state=zeros(dimension(ds)-2), diffeq...
    )
    integ = integrator(ds; diffeq...)
    idxs = SVector(idxs...)
    return basins_general(xg, yg, integ; dt, idxs, mc_att, mc_bas, mc_unmb, complete_state)
end

function basins_general(xg, yg, integ; complete_state, idxs::SVector, mc_att, mc_bas, mc_unmb, dt)
    iter_f! = (integ) -> step!(integ, dt) # we don't have to step _exactly_ `dt` here
    D = length(integ.u)
    remidxs = setdiff(1:D, idxs)
    if complete_state isa AbstractVector
        length(complete_state) ≠ D-2 && error("Vector `complete_state` must have length D-2!")
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
    bsn_nfo = draw_basin!(xg, yg, integ, iter_f!, reinit_f!, get_u, mc_att, mc_bas, mc_unmb)
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
    s[setdiff(1:D, idxs)] .= u
    reinit!(integ, s)
end


## Procedure described in  H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations, Springer, New York, 1997 Ch. 7
# The idea is to color the grid with the current color. When an attractor box is hit (even color), the initial condition is colored
# with the color of its basin (odd color). If the trajectory hits another basin 10 times in row the IC is colored with the same
# color as this basin.
function procedure!(bsn_nfo::BasinInfo, n::CartesianIndex, u, mc_att::Int, mc_bas::Int, mc_unmb::Int)
    next_c = bsn_nfo.basin[n]
    bsn_nfo.step += 1

    if iseven(next_c) && bsn_nfo.consecutive_match < mc_unmb
        # check wether or not we hit an attractor (even color). Make sure we hit mc_att consecutive times.
        if bsn_nfo.prev_attr == next_c
            bsn_nfo.prevConsecutives += 1
        else
            # reset prevConsecutives
            bsn_nfo.prev_attr = next_c
            bsn_nfo.prevConsecutives = 1
            return 0;
        end

        if bsn_nfo.prevConsecutives ≥ mc_att
            # Wait if we hit the attractor a mc_att times in a row just to check if it is not a nearby trajectory
            c3 = next_c + 1
            if mc_att == 2
                # For maps we can color the previous steps as well. Every point of the trajectory lead
                # to the attractor
                recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, c3)
            else
                # For higher dimensions we erase the past iterations and visited boxes
                recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
            end
            reset_bsn_nfo!(bsn_nfo)
            return c3
         end
    end

    if next_c == 1 && bsn_nfo.consecutive_match < mc_unmb
        # uncolored box, color it with current odd color
        bsn_nfo.basin[n] = bsn_nfo.current_color + 1
        push!(bsn_nfo.visited,n) # keep track of visited cells
        bsn_nfo.consecutive_match = 0
        return 0
    elseif next_c == 1 && bsn_nfo.consecutive_match >= mc_unmb
        # Maybe chaotic attractor, perodic or long recursion.
        # Color this box as part of an attractor
        bsn_nfo.basin[n] = bsn_nfo.current_color
        # reinit consecutive match to ensure that we have an attractor
        bsn_nfo.consecutive_match = mc_unmb
        store_attractor!(bsn_nfo, u)
        return 0
    elseif next_c == bsn_nfo.current_color + 1
        # hit a previously visited box with the current color, possible attractor?
        if bsn_nfo.consecutive_match < mc_unmb
            bsn_nfo.consecutive_match += 1
            return 0
        else
            bsn_nfo.basin[n] = bsn_nfo.current_color
            store_attractor!(bsn_nfo, u)
            # We continue iterating until we hit again the same attractor. In which case we stop.
            return 0;
        end
    elseif isodd(next_c) && 0 < next_c < bsn_nfo.current_color &&  bsn_nfo.consecutive_match < mc_unmb && mc_att == 2
        # hit a colored basin point of the wrong basin, happens all the time, we check if it happens
        # mc_bas times in a row or if it happens N times along the trajectory whether to decide if it is another basin.
        bsn_nfo.consecutive_other_basins += 1

        if bsn_nfo.prev_bas == next_c &&  bsn_nfo.prev_step == bsn_nfo.step - 1
            bsn_nfo.prevConsecutives += 1
            bsn_nfo.prev_step += 1
        else
            bsn_nfo.prev_bas = next_c
            bsn_nfo.prev_step = bsn_nfo.step
            bsn_nfo.prevConsecutives = 1
        end

        if bsn_nfo.consecutive_other_basins > 60 || bsn_nfo.prevConsecutives > mc_bas
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, next_c)
            reset_bsn_nfo!(bsn_nfo)
            return next_c
        end
        return 0
    elseif iseven(next_c) &&   (mc_unmb <= bsn_nfo.consecutive_match < 2 * mc_unmb)
        # We make sure we hit the attractor 60 consecutive times
        bsn_nfo.consecutive_match += 1
        return 0
    elseif iseven(next_c) && bsn_nfo.consecutive_match >= mc_unmb * 2
        # We have checked the presence of an attractor: tidy up everything and get a new box.
        recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
        bsn_nfo.basin[n] = bsn_nfo.current_color
        store_attractor!(bsn_nfo, u)
        # pick the next color for coloring the basin.
        bsn_nfo.current_color = bsn_nfo.next_avail_color
        bsn_nfo.next_avail_color += 2
        reset_bsn_nfo!(bsn_nfo)
        return next_c + 1;
    else
        return 0
    end
end


function store_attractor!(bsn_nfo::BasinInfo, u)
    # We divide by to order the attractors from 1 to Na
    if haskey(bsn_nfo.attractors , bsn_nfo.current_color ÷ 2)
        push!(bsn_nfo.attractors[bsn_nfo.current_color ÷ 2],  u) # store attractor
    else
        bsn_nfo.attractors[bsn_nfo.current_color ÷ 2] = Dataset([SVector(u[1],u[2])])  # init dic
    end
end

function recolor_visited_cell!(bsn_nfo::BasinInfo, old_c, new_c)
    while !isempty(bsn_nfo.visited)
       ind = pop!(bsn_nfo.visited)
       if bsn_nfo.basin[ind] == old_c
           bsn_nfo.basin[ind] = new_c
       end
    end
end

"""
    draw_basin!(xg, yg, integ, iter_f!::Function, reinit_f!::Function)
Compute an estimate of the basin of attraction on a two-dimensional plane. This is a low level function,
for higher level functions see: `basins_map2D`, `basins_general`

## Arguments
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions to test.
* `integ` : integrator handle of the dynamical system.
* `iter_f!` : function that iterates the map or the system, see step! from DifferentialEquations.jl and
examples for a Poincaré map of a continuous system.
* `reinit_f!` : function that sets the initial condition to test on a two dimensional projection of the phase space.
"""
function draw_basin!(xg, yg, integ, iter_f!::Function, reinit_f!::Function, get_u::Function, mc_att, mc_bas, mc_unmb)
    complete = false
    bsn_nfo = BasinInfo(ones(Int16, length(xg), length(yg)), xg, yg,
                iter_f!, reinit_f!, get_u, 2,4,0,0,0,1,1,0,0,Dict{Int16,Dataset{2,eltype(get_u(integ))}}(), Vector{CartesianIndex}())
    reset_bsn_nfo!(bsn_nfo)
    I = CartesianIndices(bsn_nfo.basin)
    j  = 1

    while !complete
         # pick the first empty box
         ind = 0
         for k in j:length(bsn_nfo.basin)
             if bsn_nfo.basin[I[k]] == 1
                 j = k
                 ind = I[k]
                 break
             end
         end

         if ind == 0
             # We are done
             complete = true
             break
         end

         x0 , y0  = xg[ind[1]] , yg[ind[2]]
         # Tentatively assign a color: odd is for basins, even for attractors.
         # First color is one
         bsn_nfo.basin[ind] = bsn_nfo.current_color + 1
         u0 = SVector(x0, y0)
         bsn_nfo.basin[ind] = get_color_point!(bsn_nfo, integ, u0, mc_att, mc_bas, mc_unmb)
    end
    # remove attractors and rescale from 1 to Na
    ind = iseven.(bsn_nfo.basin)
    bsn_nfo.basin[ind] .+= 1
    bsn_nfo.basin = (bsn_nfo.basin .- 1) .÷ 2
    return bsn_nfo
end

function get_color_point!(bsn_nfo::BasinInfo, integ, u0, mc_att, mc_bas, mc_unmb)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.reinit_f!(integ, u0)
    reset_bsn_nfo!(bsn_nfo)
    done = inlimbo = 0

    while done == 0
       old_u = bsn_nfo.get_u(integ)
       bsn_nfo.iter_f!(integ)
       new_u = bsn_nfo.get_u(integ)
       n = get_box(new_u, bsn_nfo)

       if !isnothing(n) # apply procedure only for boxes in the defined space
           done = procedure!(bsn_nfo, n, new_u, mc_att, mc_bas, mc_unmb)
           inlimbo = 0
       else
           # We are outside the defined grid
           inlimbo += 1
       end

       if inlimbo > 60
           done = check_outside_the_screen!(bsn_nfo, new_u, old_u, inlimbo)
       end
    end
    return done
end

function get_box(u, bsn_nfo::BasinInfo)
    xg = bsn_nfo.xg; yg = bsn_nfo.yg; # aliases
    xstep = (xg[2]-xg[1])
    ystep = (yg[2]-yg[1])
    xu, yu = u
    n = m = 0
    # check boundary
    if xu >= xg[1] && xu <= xg[end] && yu >= yg[1] && yu <= yg[end]
        n = Int(round((xu-xg[1])/xstep)) + 1
        m = Int(round((yu-yg[1])/ystep)) + 1 # +1 for 1 based indexing
    else
        return nothing
    end
    return CartesianIndex(n,m)
end

function check_outside_the_screen!(bsn_nfo::BasinInfo, new_u, old_u, inlimbo)
    if norm(new_u-old_u) < 1e-5
        recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
        reset_bsn_nfo!(bsn_nfo)
        # this CI goes to a attractor outside the screen, set to -1 (even color)
        return -1  # get next box
    elseif inlimbo > 60*20
        recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
        reset_bsn_nfo!(bsn_nfo)
        # this CI is problematic or diverges, set to -1 (even color)
        return -1  # get next box
    end
    return 0
end

function reset_bsn_nfo!(bsn_nfo::BasinInfo)
    #@show bsn_nfo.step
    bsn_nfo.consecutive_match = 0
    bsn_nfo.consecutive_other_basins = 0
    bsn_nfo.prevConsecutives = 0
    bsn_nfo.prev_attr = 1
    bsn_nfo.prev_bas = 1
    bsn_nfo.prev_step = 0
    bsn_nfo.step = 0
end


"""
    match_attractors!(b1, a1, b2, a2 [, method = :overlap])
Attempt to match the attractors in basins/attractors `b1, a1` with those at `b2, a2`.
`b, a` are expected as outputs of [`basins_map2D`](@ref) or [`basins_general`](@ref).
In these functions different attractors get assigned different IDs, however
which attractor gets which ID is somewhat arbitrary, and computing the basins of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs in `b1, a1`.

`method` decides the matching process:
* `method = :overlap` matches attractors whose basins before and after have the most
  overlap (in pixels in `b1, b2`).
* `method = :distance` matches attractors whose state space distance the smallest.
"""
function match_attractors!(b1, a1, b2, a2, method = :overlap)
    ids1, ids2 = keys(a1), keys(a2)
    if method == :overlap
        overlaps = [
            count(findall(isequal(i), b1) ∩ findall(isequal(j), ids2))
            for i in ids1, for j in ids2
        ]
    elseif method == :distance
        distances =
            [minimum()]
    end
end
end
