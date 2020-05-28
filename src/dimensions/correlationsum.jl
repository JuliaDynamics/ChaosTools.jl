using Distances
export kernelprob, correlationsum, grassberger

"""
    kernelprob(X, ε, norm = Euclidean()) → p
Associate each point in `X` (`Dataset` or timesries) with a probability `p` using the
"kernel estimation" (also called "nearest neighbor kernel estimation" and other names):
```math
p_j = \\frac{1}{N}\\sum_{i=1}^N I(||X_i - X_j|| < \\epsilon)
```
where ``N`` is its length and ``I`` gives 1 if the argument is `true`.
Because ``p`` is further normalized, it can be used as
an alternative for the [`genentropy`](@ref) function (usnig the second method).
"""
function kernelprob(X, ε, norm = Euclidean())
    N = length(X)
    p = zeros(eltype(X), N)
    @inbounds for i in 1:N
        p[i] = count(evaluate(norm, X[i], X[j]) < ε for j in 1:N)
    end
    p ./= sum(p)
    return p
end

"""
    correlationsum(X, ε, norm = Euclidean())
Calculate the correlation sum of `X` (`Dataset` or timeseries)
for a given radius `ε` and `norm`, using the formula:
```math
C(\\epsilon) = \\frac{2}{N(N-1)}\\sum_{1 ≤ i < j ≤ N} I(||X_i - X_j|| < \\epsilon)
```
where ``N`` is its length and ``I`` gives 1 if the argument is `true`.

See [`grassberger`](@ref) for more.
"""
function correlationsum(X, ε, norm = Euclidean())
    N, C = length(X), 0
    @inbounds for i in 1:N
        C += count(evaluate(norm, X[i], X[j]) < ε for j in i+1:N)
    end
    return 2C/(N*(N-1))
end

"""
    grassberger(data, εs = estimate_boxsizes(data), norm = Euclidean())
Use the method of Grassberger and Proccacia[^Grassberger1983] to estimate a correlation
dimension of the given `data`.

This function does something extrely simple:
```julia
cm = correlationsum.(Ref(data), εs, Ref(norm))
return linear_region(log.(sizes), log(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus -log(ε).
See [`generalized_dim`](@ref) for a more thorough explanation.

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function grassberger(data::AbstractDataset, εs = estimate_boxsizes(data), norm = Euclidean())
    cm = correlationsum.(Ref(data), εs, Ref(norm))
    return linear_region(log.(εs), log.(cm))[2]
end
