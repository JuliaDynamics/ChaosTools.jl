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


export molteno_dim
"""
    molteno_dim(data::Dataset; k0 = 10, q = 1.0, base = ℯ)
Calculate the generalized dimension using the algorithm for box division defined
by Molteno[^Molteno1993].

## Description
Divide the data into boxes with each new box having half the side length of the
former box using [`molteno_boxing`](@ref). Break if the number of points over
the number of filled boxes falls below `k0`. Then the generalized dimension can
be calculated by using [`genentropy`](@ref) to calculate the sum over the
logarithm also considering possible approximations and fitting this to the
logarithm of one over the boxsize using [`linear_region`](@ref).

[^Molteno1993]:
    Molteno, T. C. A., [Fast O(N) box-counting algorithm for estimating dimensions.
    Phys. Rev. E 48, R3263(R) (1993)](https://doi.org/10.1103/PhysRevE.48.R3263)
"""
function molteno_dim(data; k0 = 10, α = nothing, q=1.0, base = ℯ)
    @warn """
    `molteno_dim` is deprecated. Use instead
    ```
    probs, εs = molteno_boxing(data; k0)
    dd = genentropy.(probs; q, base)
    ````
    to get the entropies, and then `linear_region(-log.(base, εs), dd)[2]`
    to get the fractal dimension estimate.
    """
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    probs, εs = molteno_boxing(data; k0)
    dd = genentropy.(probs; q, base)
    return linear_region(-log.(base, εs), dd)[2]
end

# Remove warning for Cityblock in `lyapunov_from_data`.

export match_attractors!
function match_attractors!(args...; kwargs...)
    @warn "`match_attractors!` is deprecated in favor of `match_attractor_ids!`."
    return match_attractor_ids!(args...; kwargs...)
end