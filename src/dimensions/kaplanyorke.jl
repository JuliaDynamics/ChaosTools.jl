export kaplanyorke_dim

"""
    kaplanyorke_dim(λs::AbstractVector)
Calculate the Kaplan-Yorke dimension, a.k.a. Lyapunov dimension[^Kaplan1970].

## Description
The Kaplan-Yorke dimension is simply the point where
`cumsum(λs)` becomes zero (interpolated):
```math
 D_{KY} = k + \\frac{\\sum_{i=1}^k \\lambda_i}{|\\lambda_{k+1}|},\\quad k = \\max_j \\left[ \\sum_{i=1}^j \\lambda_i > 0 \\right].
```

If the sum of the exponents never becomes negative the function
will return the length of the input vector.

Useful in combination with [`lyapunovspectrum`](@ref).

[^Kaplan1970]: J. Kaplan & J. Yorke, *Chaotic behavior of multidimensional difference equations*, Lecture Notes in Mathematics vol. **730**, Springer (1979)
"""
function kaplanyorke_dim(v::AbstractVector)
    issorted(v, rev = true) || throw(ArgumentError(
    "The lyapunov vector must be sorted from most positive to most negative"))

    s = cumsum(v); k = length(v)
    # Find k such that sum(λ_i for i in 1:k) is still possitive
    for i in eachindex(s)
        if s[i] < 0
            k = i-1
            break
        end
    end

    if k == 0
        return zero(v[1])
    elseif k < length(v)
        return k + s[k]/abs(v[k+1])
    else
        return typeof(v[1])(length(v))
    end
end
