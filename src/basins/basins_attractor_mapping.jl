struct AttractorMapper{B<:BasinInfo, I, D, T}
    basin_info::B
    integ::I
    attractors::Dict{Int16, Dataset{D, T}}
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
