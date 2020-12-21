#=
This file contains the definition of the generalized dimension and related concepts
and also uses linear_regions.jl
=#
export boxcounting_dim, capacity_dim, generalized_dim,
       information_dim, estimate_boxsizes, kaplanyorke_dim
export molteno_dim, molteno_boxing

#######################################################################################
# Entropy based dimensions
#######################################################################################
"""
    generalized_dim(dataset [, sizes]; q = 1, base = MathConstants.e) -> D_α
Return the `α` order generalized dimension of the `dataset`, by calculating
the [`genentropy`](@ref) for each `ε ∈ sizes`.

The case of `α=0` is often called "capacity" or "box-counting" dimension.

## Description
The returned dimension is approximated by the
(inverse) power law exponent of the scaling of the [`genentropy`](@ref)
versus the box size `ε`, where `ε ∈ sizes`.

Calling this function performs a lot of automated steps:

  1. A vector of box sizes is decided by calling `sizes = estimate_boxsizes(dataset)`,
     if `sizes` is not given.
  2. For each element of `sizes` the appropriate entropy is
     calculated, through `h = genentropy.(Ref(dataset), sizes; α, base)`.
     Let `x = -log.(sizes)`.
  3. The curve `h(x)` is decomposed into linear regions,
     using [`linear_regions`](@ref)`(x, h)`.
  4. The biggest linear region is chosen, and a fit for the slope of that
     region is performed using the function [`linear_region`](@ref),
     which does a simple linear regression fit using [`linreg`](@ref).
     This slope is the return value of `generalized_dim`.

By doing these steps one by one yourself, you can adjust the keyword arguments
given to each of these function calls, refining the accuracy of the result.

The following aliases are provided:

  * α = 0 : `boxcounting_dim`, `capacity_dim`
  * α = 1 : `information_dim`
"""
function generalized_dim(α::Real, data::AbstractDataset, sizes = estimate_boxsizes(data); base = MathConstants.e)
    @warn "signature `generalized_dim(α::Real, data::Dataset, sizes)` is deprecated, use "*
          "`generalized_dim(data::Dataset, sizes; q::Real = 1.0)` instead."
    generalized_dim(data, sizes; α, base)
end
generalized_dim(α, matrix::AbstractMatrix, args...) =
generalized_dim(α, convert(AbstractDataset, matrix), args...)

function generalized_dim(data::AbstractDataset, sizes = estimate_boxsizes(data);
        α = nothing, base = Base.MathConstants.e, q = 1.0
    )
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    dd = [genentropy(data, ε; q, base) for ε ∈ sizes]
    return linear_region(-log.(base, sizes), dd)[2]
end

# Aliases, deprecated.
"capacity_dim(args...) = generalized_dim(args...; α = 0)"
function capacity_dim(args...)
    @warn "capacity_dim is deprecated."
    generalized_dim(args...; q = 0)
end

const boxcounting_dim = capacity_dim

"information_dim(args...) = generalized_dim(args...; α = 1)"
function information_dim(args...)
    @warn "information_dim is deprecated"
    generalized_dim(args...; q = 1)
end
