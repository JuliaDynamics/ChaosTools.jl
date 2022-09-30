#=
This file contains the definition of the generalized dimension and related concepts
and also uses linear_regions.jl
=#
export generalized_dim

"""
    generalized_dim(dataset [, sizes]; q = 1, base = MathConstants.e) -> Δ_q
Return the `q` order generalized dimension of the `dataset`, by calculating
the [`genentropy`](@ref) for each `ε ∈ sizes`.

The case of `q = 0` is often called "capacity" or "box-counting" dimension, while
`q = 1` is the "information" dimension.

## Description
The returned dimension is approximated by the
(inverse) power law exponent of the scaling of the [`genentropy`](@ref) ``H_q``
versus the box size `ε`, where `ε ∈ sizes`:

```math
H_q \\sim -\\Delta_q\\log(\\varepsilon)
```

Calling this function performs a lot of automated steps:

  1. A vector of box sizes is decided by calling `sizes = estimate_boxsizes(dataset)`,
     if `sizes` is not given.
  2. For each element of `sizes` the appropriate entropy is
     calculated, through `H = genentropy.(Ref(dataset), sizes; q, base)`.
     Let `x = -log.(sizes)`.
  3. The curve `H(x)` is decomposed into linear regions,
     using [`linear_regions`](@ref)`(x, h)`.
  4. The biggest linear region is chosen, and a fit for the slope of that
     region is performed using the function [`linear_region`](@ref),
     which does a simple linear regression fit using [`linreg`](@ref).
     This slope is the return value of `generalized_dim`.

By doing these steps one by one yourself, you can adjust the keyword arguments
given to each of these function calls, refining the accuracy of the result.
"""
function generalized_dim(data::AbstractDataset, sizes = estimate_boxsizes(data);
        base = Base.MathConstants.e, q = 1.0
    )
    dd = [genentropy(data, ε; q, base) for ε ∈ sizes]
    return linear_region(-log.(base, sizes), dd)[2]
end
