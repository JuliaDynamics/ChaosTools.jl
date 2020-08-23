using ChaosTools
using LinearAlgebra
using Roots
using Distances

MDI = DynamicalSystemsBase.MinimalDiscreteIntegrator
DEI = DynamicalSystemsBase.DiffEqBase.DEIntegrator

# TODO: Test with standard map, period 3, and ε large enough to cover 2 of 3 points

"""
    return_times()
Lalala

It is **crucial** that `εs` is sorted from smallest size to largest size.

The trajectory of the system is **by definition** initialized on `u0`.

Notice that this function is created with sufficiently small sets in mind, so that it is
guaranteed that the trajectory can exit the largest ε-ball in finite time.
If this doesn't happen, this algorithm will never terminate.

`εs` is always a vector. All its entries must be either real numbers or vectors (with
same size as state space dimensionality). If real numbers, each defines balls of radius `ε`.
If vectors, each defines a rectangle/box, where its length in each dimension is the entry
of the vector. Both ball/box are centered around `u0` in the state space.
**`εs` must be sorted from largest to smallest ball/box size.**

## Keywords
*

## Description
Time starts counting upon exiting the ε-ball/box and is counted until crossing again
the same ε-ball/box.
"""

# INPUT
ds = Systems.roessler()
T = 1000.0 # maximum time
u0 = trajectory(ds, 10; Ttr = 100)[end] # return center
εs = sort!([0.1, 0.01, 0.001]; rev=true)
diffeq = (;)

# INPUT
ds = Systems.standardmap()
T = 1000 # maximum time
Ttr = 10

# period 3 of standard map
u0 = SVector(0.8121, 1.6243)
# quasiperiodic around period 3:
u0 = SVector(0.964, 1.429)
εs = sort!([4.0, 0.25]; rev=true)
diffeq = (;)

# START
function check_εs_sorting(εs)
    correct = if εs[1] isa Real
        issorted(εs; rev = true)
    elseif εs[1] isa AbstractVector
        issorted(εs; by = maximum, rev = true)
    end
    if !correct
        error("`εs` must be sorted from largest to smallest ball/box size.")
    end
    return correct
end

E = length(εs)
check_εs_sorting(εs)

integ = integrator(ds, u0; diffeq...)
pre_outside = fill(false, length(εs)) # `true` if outside the ball. Previous step
cur_outside = copy(pre_outside)       # current step.
exit_times = [typeof(integ.t)[] for _ in 1:length(εs)]
entry_times = [typeof(integ.t)[] for _ in 1:length(εs)]

# Core loop
while integ.t < T
step!(integ)

# here i gives the index of the largest ε-ball that the trajectory is out of.
# It is guaranteed that the trajectory is thus outside all other boxes
i = first_outside_index(integ, u0, εs, E) # TODO: Continuous version
cur_outside[i:end] .= true
cur_outside[1:i-1] .= false

update_exit_times!(exit_times, i, pre_outside, cur_outside, integ)
update_entry_times!(entry_times, i, pre_outside, cur_outside, integ)

pre_outside .= cur_outside
end

function first_outside_index(integ::MDI, u0, εs, E)::Int
    i = findfirst(e -> isoutside(integ.u, u0, e), εs)
    return isnothing(i) ? E+1 : i
end

function update_exit_times!(exit_times, i, pre_outside, cur_outside, integ::MDI)
    @inbounds for j in i:length(pre_outside)
        cur_outside[j] && !pre_outside[j] && push!(exit_times[j], integ.t)
    end
end

function update_entry_times!(entry_times, i, pre_outside, cur_outside, integ::MDI)
    # TODO: Can I use `i` here?
    @inbounds for j in 1:length(pre_outside)
        pre_outside[j] && !cur_outside[j] && push!(entry_times[j], integ.t)
    end
end

# Here we define two functions that dispatch on the type of ε to support both balls
# and rectangles
"Return `true` if state is outside ε-ball"
function isoutside(u, u0, ε::AbstractVector)
    @inbounds for i in 1:length(u)
        abs(u[i] - u0[i]) > ε[i] && return true
    end
    return false
end
isoutside(u, u0, ε::Real) = euclidean(u, u0) > ε

"Return the **signed** distance of state to ε-ball (negative means inside ball)"
function εdistance(u, u0, ε::AbstractVector)
    m = eltype(u)(-Inf)
    @inbounds for i in 1:length(u)
        m2 = abs(u[i] - u0[i]) - ε[i]
        m2 > m && (m = m2)
    end
    return m
end
εdistance(u, u0, ε::Real) = euclidean(u, u0) - ε





# This function is only called once in the start of the algorithm, where it is certain
# that we are located at u0 and we need to record the first exit times.
function record_first_exit_times!(integ::MinimalDiscreteIntegrator, exit_times, u0, εs)
    step!(integ)
    j = length(εs)
    while any(e -> e ≤ integ.t, exit_times)
        # here i gives the index of the first ball with positive distance, i.e. the
        # largest ε-ball that the trajectory has already exited
        i = findfirst(e -> isoutside(integ.u, u0, e), εs)
        #
        # i = 0
        # for (k, ε) in enumerate(εs)
        #     if isoutside(integ.u, u0, ε)
        #         i = k
        #         break
        #     end
        # end
        if isnothing(i)
            step!(integ)
        else
            exit_times[i:j] .= integ.t
            i == 1 && return # iteration ends once largest ε is exited
            step!(integ)
            j = i
        end
    end
end


record_first_exit_times!(integ, exit_times, u0, εs)





# TODO: For continuous systems, I need two versions of this function.
# One that works as is now, and just checks at every step.
# then, I need another one that checks at every local minimum of the distance
# to the center point, and at that minimum it interpolates to see if there is
# any crossing
function step_until_in_outer_ball!(integ, u0, emax)
    tprev = integ.t
    while isoutside(integ.u, u0, emax)
        tprev = integ.t
        step!(integ)
    end
    return tprev, integ.t
end




tprev, tcurr = step_until_in_outer_ball!(integ, exit_times, u0, emax)
# I am now guaranteed that previous step was outside ε-ball, and current state is
# inside largest ε-shell.

# Here I do some simple stepping until I am outside ball, while
# also keeping track of the exit time and then the full loop starts!
# Thankfully, this function is the same for discrete and continuous systems!

# This function steps the integrator inside the balls,
# checking at every step whether a new (and smaller) ball can be
# intercepted. Helper variable already_intersected makes the
# calculations less than they have to be. The function also adds
# return times to the histogram array
function record_times_exit_εball!(integ::MinimalDiscreteIntegrator)
    # here I can make a super performant version, as no interpolation exists.
    # simply start from largest ε and then start scanning until you find ε that
    # orbit is not inside. Thus
    push!(collecte_times[end], )
end

function record_times_exit_ball!(integ::DEIntegrator)
    # Since this function is called when state is inside ball,
    # it is guaranteed that at least the outer-most ball has an
    # intersection, so use Roots and find exact number.
    f = (t) -> planecrossing(integ(t))

    # Notice that here I have to have a loop that also checks if current state
    # has already exited an ε-ball.
end
