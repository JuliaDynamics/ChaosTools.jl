DIFFEQ_DEP_WARN = """
Direct propagation of keyword arguments to DifferentialEquations.jl is deprecated.
From now on pass any DiffEq-related keywords as a `NamedTuple` using the
explicit keyword `diffeq` instead.
"""

@deprecate numericallyapunov(args...; kwargs...) lyapunov_from_data(args...; kwargs...)
@deprecate grassberger(args...; kwargs...) grassberger_dim(args...; kwargs...)

@deprecate basin_fractions basins_fractions

# Don't export `boxregion` from `sampler.jl`

export grassberger_dim
"""
    grassberger_dim(data, εs = estimate_boxsizes(data); kwargs...) → D_C
Use the method of Grassberger and Proccacia[^Grassberger1983], and the correction by
Theiler[^Theiler1986], to estimate the correlation dimension `D_C` of the given `data`.

This function does something extremely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
return linear_region(log.(sizes), log(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus log(ε).
See [`generalized_dim`](@ref) for a more thorough explanation.

See also [`takens_best_estimate`](@ref).

[^Grassberger1983]:
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)

[^Theiler1986]:
    Theiler, [Spurious dimension from correlation algorithms applied to limited time-series
    data. Physical Review A, 34](https://doi.org/10.1103/PhysRevA.34.2427)
"""
function grassberger_dim(data::AbstractDataset, εs = estimate_boxsizes(data); kwargs...)
    @warn "`grassberger_dim` is deprecated and will be removed in future versions."
    cm = correlationsum(data, εs; kwargs...)
    return linear_region(log.(εs), log.(cm))[2]
end

# Remove warning for Cityblock in `lyapunov_from_data`.