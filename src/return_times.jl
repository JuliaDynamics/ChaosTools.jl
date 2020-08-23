using ChaosTools
using LinearAlgebra
using Roots
using Distances

# TODO: What if the ε is so small that the system goes through the entire ball/box
# in one integration step...?! Shiiiiiiiiit
# ACTUALLY, doing such checks for continuous systems would lead to massive performance
# loss, because in fact any local minimum of distance from u0 would have to be checked...?

"""
    return_times()
Lalala

It is **crucial** that `εs` is sorted from smallest size to largest size.

## Keywords
*

## Description
Time starts counting upon exiting the ε-ball/box and is counted until crossing again
the same ε-ball/box.
"""

# INPUT
ds = Systems.roessler()
T = 1000.0
# This is the state that defines the ε-ball
u0 = trajectory(ds, 10; Ttr = 10)[end]
distance(x, u0) = norm(x-u0)
εs = sort!([0.01, 0.001, 0.0001])
diffeq = (;)

# START
εs isa Vector && @assert issorted(εs)
emax = εs[end]

integ = integrator(ds, u0; diffeq...)
Ttr > 0 && step!(integ, Ttr)
tprev, tcurr = integ.t, integ.t

already_intersected = fill(false, length(εs))
exit_times = fill(zero(tprev), length(εs))
collected_times = [typeof(tprev)[] for _ in 1:length(εs)]

# Here I do some simple stepping until I am outside ball, while
# also keeping track of the exit time and then the full loop starts!
# Thankfully, this function is the same for discrete and continuous systems!

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

# Here we define two functions that dispatch on the type of ε to support both balls
# and rectangles
function isoutside(u, u0, ε::AbstractVector)
    @inbounds for i in 1:length(u)
        abs(u[i] - u0[i]) > ε[i] && return true
    end
    return false
end
isoutside(u, u0, ε::Real) = euclidean(u, u0) > ε
# TODO: is this εdistance correct...? I think so.
function εdistance(u, u0, ε::AbstractVector)
    m = eltype(u)(-Inf)
    @inbounds for i in 1:length(u)
        m2 = u[i] - u0[i] - ε[i]
        m2 > m && (m = m2)
    end
    return m
end
εdistance(u, u0, ε::Real) = euclidean(u, u0) - ε

tprev, tcurr = step_until_outer_ball!(integ, u0, emax)
# I am now guaranteed that previous step was outside ε-ball, and current state is
# inside largest ε-shell.

# This function steps the integrator inside the balls,
# checking at every step whether a new (and smaller) ball can be
# intercepted. Helper variable already_intersected makes the
# calculations less than they have to be. The function also adds
# return times to the histogram array
function record_times_exit_ball!(integ::MinimalDiscreteIntegrator)
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
