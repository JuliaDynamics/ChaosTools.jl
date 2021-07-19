export kernelprob

function kernelprob(X, ε, norm)
    @warn "`kernelprob` with `norm` as positional argument is deprecated. "*
    "Use `norm` as a keyword argument instead."
    kernelprob(X, ε; norm)
end

"""
    kernelprob(X, ε; norm = Euclidean()) → p::Probabilities
Associate each point in `X` (`Dataset` or timesries) with a probability `p` using the
"kernel estimation" (also called "nearest neighbor kernel estimation" and other names):
```math
p_j = \\frac{1}{N}\\sum_{i=1}^N B(||X_i - X_j|| < \\epsilon)
```
where ``N`` is its length and ``B`` gives 1 if the argument is `true`.

See also [`genentropy`](@ref) and [`correlationsum`](@ref).
`kernelprob` is equivalent with `probabilities(X, NaiveKernel(ε, TreeDistance(norm)))`.
"""
function kernelprob(X, ε; norm = Euclidean(), w = 0)
    @warn "`kernelprob` is deprecated in favor of `probabilities(X, NaiveKernel(...))`"
    probabilities(X, NaiveKernel(ϵ; metric = norm, w))
end


"""
    permentropy_old(x::AbstractVector, order [, interval=1]; base = Base.MathConstants.e)

Compute the permutation entropy[^Brandt2002] of given `order`
from the `x` timeseries.

Optionally, `interval` can be specified to
use `x[t0:interval:t1]` when calculating permutation of the
sliding windows between `t0` and `t1 = t0 + interval * (order - 1)`.

Optionally use `base` for the logarithms.

[^Bandt2002]: C. Bandt, & B. Pompe, [Phys. Rev. Lett. **88** (17), pp 174102 (2002)](http://doi.org/10.1103/PhysRevLett.88.174102)
"""
function permentropy_old(
        time_series::AbstractArray{T, 1}, orderi::Integer,
        interval::Integer = 1;
        base=Base.MathConstants.e) where {T}

    @warn "permentropy will change to a massively faster version at the next release, and "*
          "will NOT have the `interval` keyword anymore."

    orderi > 255 && throw(ArgumentError("order = $orderi is too large, "*
              "must be smaller than $(Int(typemax(UInt8)))."))
    order = UInt8(orderi)
    # To use `searchsortedfirst`, we need each permutation to be
    # "comparable" (ordered) type.  Let's use NTuple here:
    PermType = NTuple{Int(order), UInt8}

    perms = map(PermType, permutations(1:order))
    count = zeros(UInt64, length(perms))

    for t in 1:length(time_series) - interval * order + 1
        sample = @view time_series[t:interval:t + interval * (order - 1)]
        i = searchsortedfirst(perms, PermType(sortperm(sample)))
        count[i] += 1
    end

    # To compute `p log(p)` correctly for `p = 0`, we first discard
    # cases with zero occurrence.  They don't contribute to the final
    # sum hence to the entropy:
    nonzero = [c for c in count if c != 0]

    p = nonzero ./ sum(nonzero)
    return -sum(p .* log.(base, p))
end



function non0hist(ε::Real, data::Dataset)
    @warn "signature `non0hist(ε::Real, data::Dataset)` is deprecated, use "*
          "`probabilities(data::Dataset, ε::Real)` instead."
    probabilities(data, ε)
end

function Entropies.binhist(ε::Real, data::Dataset)
    @warn "signature `binhist(ε::Real, data::Dataset)` is deprecated, use "*
          "`binhist(data::Dataset, ε::Real)` instead."
    binhist(data, ε)
end

function Entropies.genentropy(q::Real, ε::Real, data::AbstractDataset;
    base=Base.MathConstants.e)
    @warn "signature `genentropy(q::Real, ε::Real, data::Dataset)` is deprecated, use "*
          "`genentropy(data::Dataset, ε::Real; q::Real = 1.0)` instead."
    genentropy(data, ε; q, base)
end

@deprecate basins_map2D basins_2D

export basins_2D, basins_general
function basins_2D(args...; kwargs...)
    error("`basins_2D` is deprecated for `basins_of_attraction`")
end
function basins_general(args...; kwargs...)
    error("`basins_general` is deprecated for `basins_of_attraction`")
end


export non0hist, binhist, genentropy, permentropy, probabilities

"""
    permentropy(x, m = 3; τ = 1, base = Base.MathConstants.e)

Compute the permutation entropy[^Brandt2002] of given order `m`
from the `x` timeseries.

This method is textually equivalent with
```julia
genentropy(x, SymbolicPermutation(; m, τ); base)
```

[^Bandt2002]: C. Bandt, & B. Pompe, [Phys. Rev. Lett. **88** (17), pp 174102 (2002)](http://doi.org/10.1103/PhysRevLett.88.174102)
"""
function permentropy(x, m = 3; τ = 1, base = Base.MathConstants.e)
    @warn """
    The function `permentropy` is deprecated in favor of using 
    `genentropy(x, SymbolicPermutation(; m, τ); base)`.
    """
    Entropies.genentropy(x, SymbolicPermutation(; τ, m); base)
end
