#=
This file attempts to implement the "imporoved dimension estimation" by Judd:

Judd, K. (1992). An improved estimator of dimension and some comments on providing
confidence intervals. Physica D: Nonlinear Phenomena, 56(2–3), 216–228. https://doi.org/10.1016/0167-2789(92)90025-I

Judd, K. (1994). Estimating dimension from small samples. Physica D: Nonlinear Phenomena,
71(4), 421–429. https://doi.org/10.1016/0167-2789(94)90008-6

So far this attempt has failed.
=#

function interpoint_distances(X, norm = Euclidean())
    N, ds = length(X), eltype(X)[]
    sizehint!(ds, Int(N*(N-1)/2))
    @inbounds for i in 1:N
        @inbounds for j in i+1:N
            push!(ds, evaluate(norm, X[i], X[j]))
        end
    end
    return ds
end

"""
    logspace_histogram(x::Vector, λ = exp(-w0), ϵ₀ = λ^2 * maximum(x)) -> bins, counts
Calculate a logarithmically spaced histogram of the values in `x` (must be positive
definite, i.e. distances) and return the bins and count of values in the bins.

The `i`-th bin is `[ϵ₀*λ^i, ϵ₀*λ^(i+1))`. Elements of `x` greater than `ϵ₀` are disregarded.
"""
function logspace_histogram(x::AbstractVector, λ = exp(-w0(x)), ϵ₀ = λ^2 * maximum(x))
    @assert λ < 1
    bins = [ϵ₀]
    λmin = λ*minimum(x)
    λmin == 0 && error("x cannot contain zero values!")
    for i in 1:length(x)
        push!(bins, bins[end]*λ)
        bins[end] < λmin && break
    end
    reverse!(bins)
    L = length(bins)
    counts = zeros(Int, L)
    # Fill bins with values
    for v in x
        i = searchsortedfirst(bins, v)
        i == L + 1 && continue
        bins[i] += 1
    end
    return reverse!(bins), reverse!(counts)
end
# TODO: test the algorithm with including the bin [∞, ε0)
w0(x) = log(maximum(x)/minimum(x))/√length(x)

# The two papers are really complicated. The way to really do this that I will
# follow is given by the 1992 Judd paper, page 223, second column.
# p is a function ε^d * polynomial of degree `t`, with `t` " = 2 being a good choice."
# Then the formula for L (no equation number, because why write good papers) is
# "minimized" by some routine.
# Notice that pᵢ is just the function of the ϵᵢ , the bin edges.

# To estimate the dimension we have to do an optimization solve of a problem...? wtf!

p(a, bins) = @. (bins^a[1]) * (a[2] + a[3]*bins + a[4]*bins^2)
function judd_estimator(bins, count)
    f = (a) -> -sum( bins .* log.(p(a, bins)) )
    optimize(f, ones(4))
end

# The problem now is OF COURSE, that this optimization problem is ill defined.
# a polynomial can OF COURSE get negative values, and OF COURSE the logarithm
# of a negative number is not real. How do these papers get published my god...
