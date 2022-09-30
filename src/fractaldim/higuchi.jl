export higuchi

# Super duper inefficient version, but quickly coding it for now
# Also I made my own estimate for good k values that scales them logarithmically
# as we do in our fractal dimension paper. Higuchi doesn't describe this.
"""
    higuchi(x::AbstractVector [, ks])
Calculate the Higuchi dimension[^Higuchi1988] of the graph of `x`.

## Description
The Higuchi dimension is a number `Δ ∈ [1, 2]` that quantifies the roughness of
the graph of the function `x(t)`, assuming here that `x` is equi-sampled,
like in the original paper.

The method estimates how the length of the graph increases as a function of
the indices difference (which, in this context, is equivalent with differences in `t`).
Specifically, we calculate the average length versus `k` as
```math
L_m(k) = \\frac{N-1}{\\lfloor \\frac{N-m}{k} \rfloor k^2}
\\sum_{i=1}^{\\lfloor \\frac{N-m}{k} \\rfloor} |X_N(m+ik)-X_N(m+(i-1)k)| \\\\

L(k) = \\frac{1}{k} \\sum_{m=1}^k L_m(k)
```
and then use [`linear_region`](@ref) in `-log2.(k)` vs `log2.(L)` as per usual
when computing a [Fractal dimension](@ref).

The algorithm chooses default `ks` to be exponentially spaced in base-2, up to at most
exponent 8. A user can provide their own `ks` as a second argument otherwise.

Use `ChaosTools.higuchi_length(x, ks)` to obtain ``L(k)`` directly.

[^Higuchi1988]:
    Higuchi, _Approach to an irregular time series on the basis of the fractal theory_,
    [Physica D: Nonlinear Phenomena (1988)](www.doi.org/10.1016/0167-2789(88)90081-4)
"""
function higuchi(x::AbstractVector, ks = higuchi_default_ks(x))
    L = higuchi_length(x, ks)
    return linear_region(-log2.(ks), log2.(L))[2]
end

function higuchi_default_ks(x, maxpower = 8)
    exponent = floor(Int, log2(length(x))) - 1 # same as in our paper, we limit this
    exponent = min(maxpower, exponent)
    return 2 .^ (1:exponent)
end

@inbounds function higuchi_length(x::AbstractVector, ks)
    N = length(x)
    Lₘ = [zeros(k) for k in ks]
    for (j, k) in enumerate(ks)
        for m in 1:k
            Lₘ[j][m] = higuchi_inner_computation(x, m, k, N)
        end
    end
    return [(1/k)*sum(Lₘ[j][m] for m in 1:k) for (j, k) in enumerate(ks)]
end

function higuchi_inner_computation(x, m, k, N = length(x))
    a = floor(Int, (N-m)/k)
    norm = (N - 1)/a/k/k
    L = zero(eltype(x))
    @inbounds for i in 1:a
        L += abs(x[m + i*k] - x[m + (i-1)*k])
    end
    return norm*L
end