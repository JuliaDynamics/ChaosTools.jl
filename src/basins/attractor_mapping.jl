export AttractorMapper,
    AttractorsViaRecurrences,
    AttractorsViaProximity,
    AttractorsViaFeaturizing

"""
    AttractorMapper(ds::DynamicalSystem, args...; kwargs...) → mapper
Subtypes of `AttractorMapper` are structures that map initial conditions of `ds` to
attractors. Currently available mapping methods:
* [`AttractorsViaProximity`](@ref)
* [`AttractorsViaRecurrences`](@ref)
* [`AttractorsViaFeaturizing`](@ref)

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

Approximate the state space fractions `fs` of the basins of attraction of a dynamical
stystem by mapping initial conditions to attractors using `mapper`
(which contains a reference to a `ds::DynamicalSystem`), and then simply taking the
ratios of how many initial conditions ended up at each attractor.

Initial conditions to use are defined by `ics`. It can be:
* a `Dataset` of initial conditions, in which case all are used.
* a 0-argument function `ics()` that spits out random initial conditions.
  Then `N` random initial conditions are chosen.
  See [`statespace_sampler`](@ref) to generate such functions.

If `ics` is a `Dataset` then besides `fs` the `labels` of each initial condition are also
returned.

The output `fs` is a dictionary whose keys are the labels given to each attractor
(always integers enumerating the different attractors), and the
values are their respective fractions. The label `-1` is given to any initial condition
where `mapper` could not match to an attractor (this depends on the `mapper` type).
See [`AttractorMapper`](@ref) for all possible `mapper` types.

## Keyword arguments
* `N = 1000`: Number of random initial conditions to generate in case `ics` is a function.
* `show_progress = true`: Display a progress bar of the process.
"""
function basin_fractions(mapper::AttractorMapper, ics::Union{AbstractDataset, Function};
        show_progress = true, N = 1000
    )
    used_dataset = ics isa AbstractDataset
    N = used_dataset ? size(ics, 1) : N
    if show_progress
        progress=ProgressMeter.Progress(N; desc="Mapping initial conditions to attractors:")
    end
    fs = Dict{Int, Float64}()
    used_dataset && (labels = Vector{Int}(undef, N))
    # TODO: If we want to parallelize this, then we need to initialize as many
    # mappers as threads. Use a `threading` keyword and `deepcopy(mapper)`
    for i ∈ 1:N
        ic = _get_ic(ics, i)
        label = mapper(ic)
        fs[label] = get(fs, label, 0) + 1
        used_dataset && (labels[i] = label)
        show_progress && next!(progress)
    end
    ffs = Dict(k => v/N for (k, v) in fs)
    return used_dataset ? (ffs, labels) : ffs
end

_get_ic(ics::Function, i) = ics()
_get_ic(ics::AbstractDataset, i) = ics[i]

# Generic pretty printing
function generic_mapper_print(io, mapper)
    ps = 14
    text = "$(nameof(typeof(mapper)))"
    println(io, text)
    println(io, rpad(" rule f: ", ps), get_rule_for_print(mapper))
    return ps
end
Base.show(io::IO, mapper::AttractorMapper) = generic_mapper_print(io, mapper)