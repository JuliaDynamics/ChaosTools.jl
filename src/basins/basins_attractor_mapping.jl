"""
    AttractorMapper(ds; kwargs...) → mapper
Initialize a structure that maps initial conditions of `ds` to attractors.
You can call `mapper` as a function of an initial condition:
```julia
label::Int = mapper(u0)
```
and this will return the label of the attractor `u0` converges at.

`AttractorMapper` identifies attractors via recurrences. 
It has the same keywords as [`basins_of_attraction`](@ref),
so see there for how attractors are identified and what keywords you can use.
**One of keywords `grid` or `attractors` is mandatory for `AttractorMapper`.**

`AttractorMapper` has two modes of operation: **unsupervised**, with automatic detection of
attractors and **unsupervised**, where the attractors are known.
Both modes are in fact the same process that occurs in [`basins_of_attraction`](@ref),
depending on whether the `attractors` keyword is given or not.
"""
struct AttractorMapper{B<:BasinsInfo, I, K}
    bsn_nfo::B
    integ::I
    kwargs::K
end

function AttractorMapper(ds;
        # Notice that all of these are the same keywords as in `basins_of_attraction`
        grid = nothing, attractors = nothing,
        Δt=nothing, T=nothing, idxs = 1:length(grid),
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid)),
        diffeq = NamedTuple(), kwargs...
    )
    if isnothing(grid) && isnothing(attractors)
        @error "At least one of `grid` of `attractor` must be provided."
    end
    if isnothing(grid)
        # dummy grid for initialization if the second mode is used
        grid = ntuple(x -> range(-1, 1,step = 0.1), length(ds.u0))
    end

    bsn_nfo, integ = basininfo_and_integ(ds, attractors, grid, Δt, T, idxs, complete_state, diffeq)
    return AttractorMapper(bsn_nfo, integ, kwargs)
end

function (mapper::AttractorMapper)(u0; kwargs...)
    return get_label_ic!(mapper.bsn_nfo, mapper.integ, u0; mapper.kwargs...)
end


"""
ic_labelling(ds; grid = nothing, attractors = nothing, kwargs...) -> bsn_nfo, integ
This function returns a struture and an integrator that allows to match an inital condition to an
attractor. The function has two mode of operation: unsupervised with automatic detection of
attractors and unsupervised when the attractors are known. In the first mode, a grid must be defined
that contains the attractors. In the second the mode, a dictionnary that contains the attractors
must be provided with the keyword argument `attractors`.

## Keyword Arguments
* `grid`:  tuple of ranges defining the grid of initial conditions.
* `attractors`: a dictionary with keys corresponding to the number of the attractor.
* `kwargs` : The interface is the same as [`basins_of_attraction`](@ref) and has the
same keyword arguments except for the keywords controlling the basins estimation.

The initial condition `u0` is matched to an attractor with the function [`get_label_ic!`](@ref).

Example:
```
ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
u0 = [1., 1.]
xg = yg = range(-2.,2.,length = 100)
bsn_nfo, integ = ic_labelling((xg,yg), ds)
label = get_label_ic!(bsn_nfo, integ, u0)
```
"""
function ic_labelling(ds; grid = nothing, attractors = nothing, kwargs...)
    if isnothing(grid) && isnothing(attractors)
        @error "At least one of the two keyword `grid` of `attractor` must be provided"
    end

    if isnothing(grid)
        # dummy grid for initialization if the second mode is used
        grid = ntuple(x -> range(-1, 1,step = 0.1), length(ds.u0))
    end
    return basins_of_attraction(grid, ds; ic_lab_mode = true, attractors, kwargs...)
end
