
#######################################################################################
# Correlation-sum based
#######################################################################################
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
an alternative for the [`genentropy`](@ref) function (using the second method).
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
    correlationsum(X, ε::Real; w = 1, norm = Euclidean()) → C(ε)
Calculate the correlation sum of `X` (`Dataset` or timeseries)
for a given radius `ε` and `norm`, using the formula:
```math
C(\\epsilon) = \\frac{2}{(N-w)(N-w-1)}\\sum_{i=1}^{N}\\sum_{j=1+w+i}^{N} I(||X_i - X_j|| < \\epsilon)
```
where ``N`` is its length and ``I`` gives 1 if the argument is `true`.
`w` is the Theiler window, a correction to the correlation sum that skips points
that are temporally close with each other, with the aim of removing spurious correlations.

See the book "Nonlinear Time Series Analysis", Ch. 6, for a discussion
around `w` and choosing best values.

See [`grassberger`](@ref) for more.
"""
function correlationsum(X, ε::Real; norm = Euclidean(), w = 1)
    N, C = length(X), 0
    @inbounds for i in 1:N
        C += count(evaluate(norm, X[i], X[j]) < ε for j in i+1+w:N)
    end
    return 2C/((N-w)*(N-1-w))
end

"""
    correlationsum(X, εs::AbstractVector; kwargs...) → Cs
Calculate the correlation sum for every `ε ∈ εs` using an optimized version.
"""
function correlationsum(X, εs::AbstractVector; norm = Euclidean(), w = 1)
    @assert issorted(εs) "Sorted εs required for optimized version."
    d = distancematrix(X, norm)
    Cs = zeros(length(εs))
    N = length(X)
    factor = 2/((N-w)*(N-1-w))
    for k in length(εs)÷2:-1:1
        ε = εs[k]
        for i in 1:N
            @inbounds Cs[k] += count(d[j, i] < ε for j in i+1+w:N)
        end
        Cs[k] == 0 && break
    end
    for k in (length(εs)÷2 + 1):length(εs)
        ε = εs[k]
        for i in 1:N
            @inbounds Cs[k] += count(d[j, i] < ε for j in i+1+w:N)
        end
        if Cs[k] ≈ 1/factor
            Cs[k:end] .= 1/factor
            break
        end
    end
    return Cs .* factor
end

function distancematrix(X, norm = Euclidean())
    N = length(X)
    d = zeros(eltype(X), N, N)
    @inbounds for i in 1:N
        for j in i+1:N
            d[j, i] = evaluate(norm, X[i], X[j])
        end
    end
    return d
end

"""
    grassberger(data, εs = estimate_boxsizes(data); kwargs...)
Use the method of Grassberger and Proccacia[^Grassberger1983], and the correction by
Theiler[^Theiler1986], to estimate the correlation dimension of the given `data`.

This function does something extrely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
return linear_region(log.(sizes), log(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus -log(ε).
See [`generalized_dim`](@ref) for a more thorough explanation.

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)

[^Theiler1986]: Theiler, [Spurious dimension from correlation algorithms applied to limited time-series data. Physical Review A, 34](https://doi.org/10.1103/PhysRevA.34.2427)
"""
function grassberger(data::AbstractDataset, εs = estimate_boxsizes(data); kwargs...)
    cm = correlationsum(data, εs; kwargs...)
    return linear_region(log.(εs), log.(cm))[2]
end
