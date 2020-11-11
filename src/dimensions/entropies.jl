#=
The code that used to be on this document has been moved to Entropies.jl
and improved in its entirety. This page now only holds deprecations.
=#
using Entropies
export non0hist, binhist, genentropy, permentropy, probabilities

using Combinatorics: permutations


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

function Entropies.genentropy(α::Real, ε::Real, data::AbstractDataset;
    base=Base.MathConstants.e)
    @warn "signature `genentropy(α::Real, ε::Real, data::Dataset)` is deprecated, use "*
          "`genentropy(data::Dataset, ε::Real; α::Real = 1.0)` instead."
    genentropy(data, ε; α, base)
end



"""
    permentropy(x::AbstractVector, order [, interval=1]; base = Base.MathConstants.e)

Compute the permutation entropy[^Brandt2002] of given `order`
from the `x` timeseries.

Optionally, `interval` can be specified to
use `x[t0:interval:t1]` when calculating permutation of the
sliding windows between `t0` and `t1 = t0 + interval * (order - 1)`.

Optionally use `base` for the logarithms.

[^Bandt2002]: C. Bandt, & B. Pompe, [Phys. Rev. Lett. **88** (17), pp 174102 (2002)](http://doi.org/10.1103/PhysRevLett.88.174102)
"""
function permentropy(
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


function permentropy2()
    Entropies.genentropy(x, SymbolicPermutation(), τ = 1, m = 4)
end
