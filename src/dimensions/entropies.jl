using Combinatorics: permutations

export non0hist, genentropy, permentropy

"""
```julia
non0hist(ε, dataset::AbstractDataset)
```
Partition a dataset into tabulated intervals (boxes) of
size `ε` and return the sum-normalized histogram in an unordered 1D form,
discarding all zero elements.

## Performances Notes
This method is effecient in both memory
and speed, because it uses a dictionary to collect the information of bins with
elements, while it completely disregards empty bins. This allows
computation of entropies of high-dimensional datasets and
with small box sizes `ε` without memory overflow.

Use e.g. `fit(Histogram, ...)` from
[`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/) if you
wish to keep information about the edges of the binning as well
as the zero elements.
"""
function non0hist(ε::Real, data::AbstractDataset{D, T}) where {D, T<:Real}
    # Initialize:
    mini = minima(data)
    L = length(data)
    # Perform "Histogram":
    # `d` is a dictionary that contains all the histogram information
    # the keys are the bin edges indices and the values are the amount of
    # datapoints in each bin
    d = Dict{SVector{D, Int}, Int}()
    for point in data
        # index of datapoint in the ranges space:
        # It is necessary to convert Floor to Int (representation issues)
        ind::SVector{D, Int} = floor.(Int, (point - mini)/ε)

        # Add 1 to the bin that contains the datapoint:
        haskey(d, ind) || (d[ind] = 0) #check if you need to create key (= bin)
        d[ind] += 1
    end
    return collect(values(d))./L
end

non0hist(ε::Real, matrix) = non0hist(ε, convert(Dataset, matrix))



"""
```julia
genentropy(α, ε, dataset::AbstractDataset; base = e)
```
Compute the `α` order generalized (Rényi) entropy [1] of a dataset,
by first partitioning it into boxes of length `ε` using [`non0hist`](@ref).
```julia
genentropy(α, p::AbstractArray; base = e)
```
Compute the entropy of an array `p` directly, assuming that `p` is
sum-normalized.

Optionally use `base` for the logarithms.

## Description
Let ``p`` be an array of probabilities (summing to 1). Then the Rényi entropy is
```math
R_\\alpha(p) = \\frac{1}{1-\\alpha}\\sum_i p[i]^\\alpha
```
and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [2]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

## References

[1] : A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics,
Statistics and Probability*, pp 547 (1960)

[2] : C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function genentropy(α::Real, ε::Real, data::AbstractDataset; base=Base.e)
    ε < 0 && throw(ArgumentError("Box-size for entropy calculation must be ≥ 0."))
    p = non0hist(ε, data)
    return genentropy(α, p; base = base)
end
genentropy(α::Real, ε::Real, matrix; base = e) =
genentropy(α, ε, convert(Dataset, matrix); base = base)

function genentropy{T<:Real}(α::Real, p::AbstractArray{T}; base = e)
  α < 0 && throw(ArgumentError("Order of Rényi entropy must be ≥ 0."))

  if α ≈ 0
    return log(base, length(p)) #Hartley entropy, max-entropy
  elseif α ≈ 1
    return -sum( x*log(base, x) for x in p ) #Shannon entropy, information to locate with ε
  elseif isinf(α)
    return -log(base, maximum(p)) #Min entropy
  else
    return (1/(1-α))*log(base, sum(x^α for x in p) ) #Renyi α entropy
  end
end



"""
    permentropy(x::AbstractVector, order [, interval=1]; base = e)

Compute the permutation entropy [1] of given `order`
from the `x` timeseries.

Optionally, `interval` can be specified to
use `x[t0:interval:t1]` when calculating permutation of the
sliding windows between `t0` and `t1 = t0 + interval * (order - 1)`.

Optionally use `base` for the logarithms.

## References

[1] : C. Bandt, & B. Pompe, [Phys. Rev. Lett. **88** (17), pp 174102 (2002)](http://doi.org/10.1103/PhysRevLett.88.174102)
"""
function permentropy(
        time_series::AbstractArray{T, 1}, order::UInt8,
        interval::Integer = 1;
        base=Base.e) where {T}

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
    return - sum(p .* log.(base, p))
end

function permentropy(time_series, order::Integer, args...; kwargs...)
    order = try
        UInt8(order)
    catch err
        if isa(err, InexactError)
            error("order = $order is too large.",
                  " order smaller than $(Int(typemax(UInt8))) can be used.")
        else
            rethrow()
        end
    end
    return permentropy(time_series, order, args...; kwargs...)
end
