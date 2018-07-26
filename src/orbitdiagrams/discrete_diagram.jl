export orbitdiagram

"""
    orbitdiagram(ds::DiscreteDynamicalSystem, i, p_index, pvalues; kwargs...)
Compute the orbit diagram (also called bifurcation diagram) of the given system
for the `i`-th variable for parameter values `pvalues`. The `p_index` specifies
which parameter of the equations of motion is to be changed.

Returns a vector of vectors, where each entry are the points are each parameter value.

## Keyword Arguments
* `ics = [get_state(ds)]` : container of initial conditions that
  are used at each parameter value to evolve orbits.
* `Ttr::Int = 1000` : Transient steps;
  each orbit is evolved for `Ttr` first before saving output.
* `n::Int = 100` : Amount of points to save for each initial condition.

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(ds::DDS{IIP, S, D}, i::Int, p_index, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [get_state(ds)]) where {IIP, S, D}

    p0 = ds.p[p_index]
    if D == 1
        i != 1 &&
        error("You have a 1D system and yet you gave `i=$i`. What's up with that!?")
    end

    output = [zeros(eltype(S), n*length(ics)) for i in 1:length(pvalues)]

    integ = integrator(ds)

    for (j, p) in enumerate(pvalues)

        integ.p[p_index] = p

        for (m, st) in enumerate(ics)
            reinit!(integ, st)
            step!(integ, Ttr)

            for k in 1:n
                step!(integ)
                output[j][(m-1)*n + k] = integ.u[i]
            end
        end
    end
    ds.p[p_index] = p0
    return output
end
