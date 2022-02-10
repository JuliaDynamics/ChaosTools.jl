"""
    AttractorMapper(ds::DynamicalSystem, args...; kwargs...) → mapper
Subtypes of `AttractorMapper` are structures that map initial conditions of `ds` to 
attractors. Currently available mapping methods:
* [`AttractorsViaProximity`](@ref)
* [`AttractorsViaRecurrences`](@ref)

`AttractorMapper` subtypes can always be used directly with [`basin_fractions`](@ref).

In addition, some mappers can be called as a function of an initial condition:
```julia
label = mapper(u0)
```
and this will on the fly compute and return the label of the attractor `u0` converges at.
The mappers that can do this are:
* [`AttractorsViaProximity`](@ref)
* [`AttractorsViaRecurrences`](@ref)
"""
abstract type AttractorMapper end

# Generic method for `basin_fractions` here
function basin_fractions(mapper::AttractorMapper, ics::Union{Dataset, Function};
        show_progress = true, N = 1000
    )
    N = (typeof(ics) <: Function)  ? N : size(ics, 1) # number of actual ICs
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:")
    end
    fs = Dict{Int, Float64}()
    Threads.@threads for i = 1:N
        ic = _get_ic(ics, i)
        label = mapper(ic)
        fs[label] += 1
        show_progress && next!(progress)
    end
    return Dict(k => v/N for (k, v) in fs)
end

_get_ic(ics::Function, i) = ics()
_get_ic(ics::Dataset, i) = ics[i]