using DynamicalSystems
using LinearAlgebra

# INPUT
ds = Systems.roessler()
T = 1000.0
# This is the state that defines the ε-ball
u0 = trajectory(ds, 10; Ttr = 10)[end]
distance(x, u0) = norm(x-u0)
es = sort!([0.01, 0.001, 0.0001])
diffeq = (;)


# START
@assert issorted(es)
emax = es[end]

# Providing a custom distance function allows elliptic
# distance weighting, in case e.g. one of the variables
# has fundamentally larger timescale.

integ = integrator(ds, u0; diffeq...)
tprev, tcurr = integ.t, integ.t

already_intersected = fill(false, length(es))
exit_times = fill(zero(tprev), length(es))
collected_times = [typeof(tprev)[] for _ in 1:length(es)]

# Here I do some simple stepping until I am outside ball, while
# also keeping track of the exit time and then the full loop starts!

function step_until_outer_ball!(integ, u0, emax)
    tprev = integ.t
    while distance(integ.u, u0) > emax
        tprev = integ.t
        step!(integ)
    end
    return tprev, integ.t
end

tprev, tcurr = step_until_outer_ball!(integ, u0, emax)

# This function steps the integrator inside the balls,
# checking at every step whether a new (and smaller) ball can be
# intercepted. Helper variable already_intersected makes the
# calculations less than they have to be. The function also adds
# return times to the histogram array
function step_while_inside_ball!()
    # Since this function is called when state is inside ball,
    # it is guaranteed that at least the outer-most ball has an
    # intersection, so use Roots and find exact number.
end
