export draw_basin!, basin_map, basin_general_ds


mutable struct BasinInfo{I,F,V}
    basin :: I
    xg :: F
    yg :: F
    iter_f! :: Function
    reinit_f! :: Function
    get_u :: Function
    current_color :: Int64
    next_avail_color :: Int64
    consecutive_match :: Int64
    consecutive_other_basins :: Int64
    prevConsecutives :: Int64
    prev_attr :: Int64
    prev_bas :: Int64
    prev_step :: Int64
    step :: Int64
    attractors :: V
end

function Base.show(io::IO, bsn::BasinInfo)
    println(io, "Basin of attraction structure")
    println(io,  rpad(" size : ", 14),    size(bsn.basin))
    println(io,  rpad(" Number of attractors found: ", 14),   Int((bsn.current_color-2)/2)  )
end


"""
    basin_map(xg, yg, integ; kwargs...)
Compute an estimate of the basin of attraction on a two-dimensional plane using a map of the plane onto itself.
The dynamical system should be a discrete two dimensional system such as:
    * Discrete 2D map.
    * 2D poincaré map.
    * A 2D stroboscopic map.
    * A dynamical system with a projection on the plane

[H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations, Springer, New York, 1997]

## Arguments
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions to test.
* `integ` : A  integrator handle of the dynamical system. For a Poincaré map, the handle is a `pmap`
as defined in [`poincaremap`](@ref)

## Keyword Arguments
* `T` : Period of the stroboscopic map.
* `Ncheck` : A parameter that sets the number of consecutives hits of an attractor before deciding the basin
of the initial condition.


## Example:

```jl
using DynamicalSystems, ChaosTools
ds = Systems.rikitake(μ = 0.47, α = 1.0)
xg=range(-6.,6.,length=150); yg=range(-6.,6.,length=150)
pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)
bsn = basin_map(xg, yg, pmap)
```

"""
function basin_map(xg, yg, pmap::PoincareMap; Ncheck = 2)
    reinit_f! = (pmap,y) -> _init_map(pmap, y, pmap.i)
    get_u = (pmap) -> pmap.integ.u[pmap.i]
    basin = draw_basin!(xg, yg, pmap, step!, reinit_f!, get_u, Ncheck)
end


function _init_map(pmap::PoincareMap, y, idxs)
    u = zeros(1,length(pmap.integ.u))
    u[idxs] = y
    # all other coordinates are zero
    reinit!(pmap, u)
end


function basin_map(xg, yg, integ; T=0., Ncheck = 2)
    if T>0
        iter_f! = (integ) -> step!(integ, T, true)
    else
        iter_f! = (integ) -> step!(integ)
    end
    reinit_f! =  (integ,y) -> reinit!(integ, y)
    get_u = (integ) -> integ.u

    return draw_basin!(xg, yg, integ, iter_f!, reinit_f!, get_u, Ncheck)
end






"""
    basin_general(xg, yg, integ; T=1., idxs=1:2)
Compute an estimate of the basin of attraction on a two-dimensional plane using a stroboscopic map.

## Arguments
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions to test.
* `integ` : integrator handle of the dynamical system.

## Keyword Arguments
* `dt` : Time step of the integrator. It recommended to use values above 1.
* `idxs = 1:D` : Optionally you can choose which variables to save. Defaults to the entire state.
* `Ncheck` : A parameter that sets the number of consecutives hits of an attractor before deciding the basin
of the initial condition.

## Example:
```jl
using DynamicalSystems, ChaosTools
ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
integ = integrator(ds, u0=[0,0,0,0], reltol=1e-9)
xg=range(-4,4,length=150)
yg=range(-4,4,length=150)
bsn = basin_general_ds(xg, yg, integ; dt=1., idxs=1:2)
"""
function basin_general_ds(xg, yg, integ; dt=1., idxs=1:2, Ncheck = 10)
    i = typeof(idxs) <: Int ? i : SVector{length(idxs), Int}(idxs...)
    iter_f! = (integ) -> step!(integ, dt, true)
    reinit_f! =  (integ,y) -> _init_ds(integ, y, i)
    get_u = (integ) -> integ.u[i]
    return draw_basin!(xg, yg, integ, iter_f!, reinit_f!,get_u, Ncheck)
end


function _init_ds(integ, y, idxs)
    u = zeros(length(integ.u))
    u[idxs] = y
    # all other coordinates are zero
    reinit!(integ, u)
end


## Procedure described in  H. E. Nusse and J. A. Yorke, Dynamics: numerical explorations, Springer, New York, 2012
# The idea is to color the grid with the current color. When an attractor box is hit (even color), the initial condition is colored
# with the color of its basin (odd color). If the trajectory hits another basin 10 times in row the IC is colored with the same
# color as this basin.
function procedure!(bsn_nfo::BasinInfo, n::Int, m::Int, u, Ncheck::Int;  max_check = 60)
    next_c = bsn_nfo.basin[n,m]
    bsn_nfo.step += 1


    if iseven(next_c) && bsn_nfo.consecutive_match < max_check
        # check wether or not we hit an attractor (even color). Make sure we hit Ncheck consecutive times.
        if bsn_nfo.prev_attr == next_c
            bsn_nfo.prevConsecutives +=1
        else
            # reset prevConsecutives
            bsn_nfo.prev_attr = next_c
            bsn_nfo.prevConsecutives =1
            return 0;
        end

        if bsn_nfo.prevConsecutives ≥ Ncheck
            # Wait if we hit the attractor a Ncheck times in a row just to check if it is not a nearby trajectory
            #println("found IC")
            c3 = next_c+1
            ind = findall(bsn_nfo.basin .== bsn_nfo.current_color+1)
            if Ncheck == 2
                # For maps we can color the previous steps as well. Every point of the trajectory lead
                # to the attractor
                for k in ind; bsn_nfo.basin[k] = c3; end
            else
                # For higher dimensions we erase the past iterations.
                # erase visited boxes
                for k in ind; bsn_nfo.basin[k] = 1; end
            end
            reset_bsn_nfo!(bsn_nfo)
            return c3
         end
    end

    if next_c == 1 && bsn_nfo.consecutive_match < max_check
        # uncolored box, color it with current odd color
        bsn_nfo.basin[n,m] = bsn_nfo.current_color + 1
        bsn_nfo.consecutive_match = 0
        return 0
    elseif next_c == 1 && bsn_nfo.consecutive_match >= max_check
        # Maybe chaotic attractor, perodic or long recursion.
        # Color this box as part of an attractor
        bsn_nfo.basin[n,m] = bsn_nfo.current_color
        # reinit consecutive match to ensure that we have an attractor
        bsn_nfo.consecutive_match = max_check
        push!(bsn_nfo.attractors, [bsn_nfo.current_color/2, u...]) # store attractor
        return 0
    elseif next_c == bsn_nfo.current_color + 1
        # hit a previously visited box with the current color, possible attractor?
        if bsn_nfo.consecutive_match < max_check
            bsn_nfo.consecutive_match += 1
            return 0
        else
            bsn_nfo.basin[n,m] = bsn_nfo.current_color
            push!(bsn_nfo.attractors, [bsn_nfo.current_color/2, u...]) # store attractor
            # We continue iterating until we hit again the same attractor. In which case we stop.
            return 0;
        end
    elseif isodd(next_c) && 0 < next_c < bsn_nfo.current_color &&  bsn_nfo.consecutive_match < max_check && Ncheck == 2
        # hit a colored basin point of the wrong basin, happens all the time, we check if it happens
        #10 times in a row or if it happens N times along the trajectory whether to decide if it is another basin.
        bsn_nfo.consecutive_other_basins += 1

        if bsn_nfo.prev_bas == next_c &&  bsn_nfo.prev_step == bsn_nfo.step-1
            bsn_nfo.prevConsecutives +=1
            bsn_nfo.prev_step += 1
        else
            bsn_nfo.prev_bas = next_c
            bsn_nfo.prev_step = bsn_nfo.step
            bsn_nfo.prevConsecutives =1
        end

        if bsn_nfo.consecutive_other_basins > 60 || bsn_nfo.prevConsecutives > 10
            ind = findall(bsn_nfo.basin .== bsn_nfo.current_color+1)
            for k in ind; bsn_nfo.basin[k] = next_c; end

            reset_bsn_nfo!(bsn_nfo)
            return next_c
        end
        return 0
    elseif iseven(next_c) &&   (max_check <= bsn_nfo.consecutive_match < 2*max_check)
        # We make sure we hit the attractor 60 consecutive times
        bsn_nfo.consecutive_match+=1
        return 0
    elseif iseven(next_c) && bsn_nfo.consecutive_match >= max_check*2
        # We have checked the presence of an attractor: tidy up everything and get a new box.
        ind = findall(bsn_nfo.basin .== bsn_nfo.current_color+1)
        for k in ind; bsn_nfo.basin[k] = 1; end

        bsn_nfo.basin[n,m] = bsn_nfo.current_color
        push!(bsn_nfo.attractors, [bsn_nfo.current_color/2, u...]) # store attractor

        # pick the next color for coloring the basin.
        bsn_nfo.current_color = bsn_nfo.next_avail_color
        bsn_nfo.next_avail_color += 2

        reset_bsn_nfo!(bsn_nfo)
        return next_c+1;
    else
        return 0
    end
end


"""
    draw_basin!(xg, yg, integ, iter_f!::Function, reinit_f!::Function)
Compute an estimate of the basin of attraction on a two-dimensional plane. This is a low level function,
for higher level functions see: `basin_map`, `basin_general_ds`

## Arguments
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions to test.
* `integ` : integrator handle of the dynamical system.
* `iter_f!` : function that iterates the map or the system, see step! from DifferentialEquations.jl and
examples for a Poincaré map of a continuous system.
* `reinit_f!` : function that sets the initial condition to test on a two dimensional projection of the phase space.
"""
function draw_basin!(xg, yg, integ, iter_f!::Function, reinit_f!::Function, get_u::Function, Ncheck)

    complete = 0;

    bsn_nfo = BasinInfo(ones(Int16, length(xg), length(yg)), xg, yg, iter_f!, reinit_f!, get_u, 2,4,0,0,0,1,1,0,0,[])

    reset_bsn_nfo!(bsn_nfo)

    while complete == 0
         # pick the first empty box
         get_empt_box = findall(bsn_nfo.basin .== 1)
         if length(get_empt_box) == 0
             complete = 1
             break
         end

         ni = get_empt_box[1][1]; mi = get_empt_box[1][2]
         x0 = xg[ni]; y0 = yg[mi]

         # Tentatively assign a color: odd is for basins, even for attractors.
         # First color is one
         bsn_nfo.basin[ni,mi] = bsn_nfo.current_color + 1

         u0=[x0, y0]

         bsn_nfo.basin[ni,mi] = get_color_point!(bsn_nfo, integ, u0; Ncheck=Ncheck)
    end

    return bsn_nfo
end



function get_color_point!(bsn_nfo::BasinInfo, integ, u0; Ncheck=2)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.reinit_f!(integ, u0)
    reset_bsn_nfo!(bsn_nfo)

    done = 0;
    inlimbo = 0

    while done == 0
       old_u = bsn_nfo.get_u(integ)
       bsn_nfo.iter_f!(integ)
       new_u = bsn_nfo.get_u(integ)

       n,m = get_box(new_u, bsn_nfo)

       if n>=0 # apply procedure only for boxes in the defined space
           done = procedure!(bsn_nfo, n, m, new_u, Ncheck)
           inlimbo = 0
       else
           # We are outside the defined grid
           inlimbo +=1
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

    xu=u[1]
    yu=u[2]
    n=0; m=0;
    # check boundary
    if xu >= xg[1] && xu <= xg[end] && yu >= yg[1] && yu <= yg[end]
        n = Int(round((xu-xg[1])/xstep))+1
        m = Int(round((yu-yg[1])/ystep))+1 # +1 for 1 based indexing
    else
        n=-1
        m=-1
    end
    return n,m
end


function check_outside_the_screen!(bsn_nfo::BasinInfo, new_u, old_u, inlimbo)

    if norm(new_u-old_u) < 1e-5
        #println("Got stuck somewhere, Maybe an attractor outside the screen: ", new_u)
        ind = findall(bsn_nfo.basin .== bsn_nfo.current_color+1)
        [ bsn_nfo.basin[k[1],k[2]] = 1  for k in ind]
        reset_bsn_nfo!(bsn_nfo)
        # this CI goes to a attractor outside the screen, set to -1 (even color)
        return -1  # get next box
    elseif inlimbo > 60*20
        #println("trajectory diverges: ", new_u)
        ind = findall(bsn_nfo.basin .== bsn_nfo.current_color+1)
        [ bsn_nfo.basin[k[1],k[2]] = 1  for k in ind]
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
