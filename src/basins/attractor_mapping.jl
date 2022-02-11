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

# Generic method for `basin_fractions` here.
# It works for all mappers that define the function-like-object behavior
"""
    basin_fractions(mapper::AttractorMapper, ics::Union{Dataset, Function}; kwargs...)

Compute the state space fractions `fs` of the basins of attraction of the given dynamical
system by sampling initial conditions from `ics`, mapping them to attractors using `mapper`
(which contains a reference to a `ds::DynamicalSystem`), and then simply taking the 
ratios of how many initial conditions ended up to each attractor.

The output `fs` is a dictionary whose keys are the labels given to each attractor, and the
values are their respective fractions. The label `-1` is given to any initial condition
which did not match any of the known attractors of `mapper`. See [`AttractorMapper`](@ref)
for all possible `mapper` types.

If `ics` is a `Dataset`, besides `fs` the `labels` of each initial condition are also
returned.

## Keyword arguments
* `N=1000`: Number of sample initial conditions to generate in case `ics` is a function.
* `show_progress = true`: Display a progress bar of the process.

## Parallelization note
The trajectories in this method are integrated in parallel using `Threads`.
To enable this, simply start Julia with the number of threads you want to use.

[^Stender2021] : Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function basin_fractions(mapper::AttractorMapper, ics::Union{AbstractDataset, Function};
        show_progress = true, N = 1000
    )
    N = (ics isa Function) ? N : size(ics, 1) # number of actual ICs
    if show_progress
        progress=ProgressMeter.Progress(N; desc="Mapping initial conditions to attractors:")
    end
    fs = Dict{Int, Float64}()
    # TODO: If we want to parallelize this, then we need to initialize as many
    # mappers as threads. Use a `threading` keyword and `deepcopy(mapper)`
    for i ∈ 1:N
        ic = _get_ic(ics, i)
        label = mapper(ic)
        fs[label] = get(fs, label, 0) + 1
        show_progress && next!(progress)
    end
    return Dict(k => v/N for (k, v) in fs)
end

_get_ic(ics::Function, i) = ics()
_get_ic(ics::Dataset, i) = ics[i]