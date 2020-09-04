#######################################################################################
# Original correlation sum
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
See also [`takens_best_estimate`](@ref).
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
    grassberger(data, εs = estimate_boxsizes(data); kwargs...) → D_C
Use the method of Grassberger and Proccacia[^Grassberger1983], and the correction by
Theiler[^Theiler1986], to estimate the correlation dimension `D_C` of the given `data`.

This function does something extrely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
return linear_region(log.(sizes), log(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus log(ε).
See [`generalized_dim`](@ref) for a more thorough explanation.

See also [`takens_best_estimate`](@ref).

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)

[^Theiler1986]: Theiler, [Spurious dimension from correlation algorithms applied to limited time-series data. Physical Review A, 34](https://doi.org/10.1103/PhysRevA.34.2427)
"""
function grassberger(data::AbstractDataset, εs = estimate_boxsizes(data); kwargs...)
    cm = correlationsum(data, εs; kwargs...)
    return linear_region(log.(εs), log.(cm))[2]
end


#######################################################################################
# Takens' best estimate
#######################################################################################
export takens_best_estimate

"""
    takens_best_estimate(X, εmax, metric = Chebyshev(),εmin = 0) → D_C
Use the so-called "Takens' best estimate" [^Takens1985][^Theiler1988]
method for estimating the correlation dimension
`D_C` for the given dataset `X`.

The original formula is
```math
D_C \\approx \\frac{C(\\epsilon_\\text{max})}{\\int_0^{\\epsilon_\\text{max}}(C(\\epsilon) / \\epsilon) \\, d\\epsilon}
```
where ``C`` is the [`correlationsum`](@ref) and ``\\epsilon_\\text{max}`` is an upper cutoff.
Here we use the later expression
```math
D_C \\approx - \\frac{1}{\\eta},\\quad \\eta = \\frac{1}{(N-1)^*}\\sum_{[i, j]^*}\\log(||X_i - X_j|| / \\epsilon_\\text{max})
```
where the sum happens for all ``i, j`` so that ``i < j`` and ``||X_i - X_j|| < \\epsilon_\\text{max}``. In the above expression, the bias in the original paper has already been corrected, as suggested in [^Borovkova1999].

If `X` comes from a delay coordinates embedding of a timseries `x`, a recommended value
for ``\\epsilon_\\text{max}`` is `std(x)/4`.

[^Takens1985]: Takens, On the numerical determination of the dimension of an attractor, in: B.H.W. Braaksma, B.L.J.F. Takens (Eds.), Dynamical Systems and Bifurcations, in: Lecture Notes in Mathematics, Springer, Berlin, 1985, pp. 99–106.
[^Theiler1988]: Theiler, [Lacunarity in a best estimator of fractal dimension. Physics Letters A, 133(4–5)](https://doi.org/10.1016/0375-9601(88)91016-X)
[^Borovkova1999]: Borovkova et al., [Consistency of the Takens estimator for the correlation dimension. The Annals of Applied Probability, 9, 05 1999.](https://doi.org/10.1214/aoap/1029962747)
"""
function takens_best_estimate(X, εmax, metric = Chebyshev(), εmin=0)
    n, η, N = 0, zero(eltype(X)), length(X)
    @inbounds for i in 1:N
        for j in i+1:N
            d = evaluate(metric, X[i], X[j])
            if εmin < d < εmax
                n += 1
                η += log(d/εmax)
            end
        end
    end
    return -(n-1)/η
end

"""
    shirer_estimate(X, εmax, metric = Chebyshev(),εmin = 0) → D_C
Use the extension by Shirer [^Shirer1997] of the so-called "Takens' best estimate" [^Takens1985][^Theiler1988]
method for estimating the correlation dimension
`D_C` for the given dataset `X`.

The formula depends on `p` and `k` and is given by
```math
D_{pk}(ε_0) \\approx  \\frac{\\sum_{r_i<\\varepsilon_0}\\left(\\frac{r_i}{\\varepsilon_0}\\right)^p\\bigg\\vert\\ln\\left(\\frac{r_i}{\\varepsilon_0}\\right)\\bigg\\vert^{k-1}}{\\sum_{r_i<\\varepsilon_0}\\left(\\frac{r_i}{\\varepsilon_0}\\right)^p\\bigg\\vert\\ln\\left(\\frac{r_i}{\\varepsilon_0}\\right)\\bigg\\vert^{k}}-p
```
where the sum happens for all ``i, j`` so that ``i < j`` and ``||X_i - X_j|| < \\epsilon_\\text{max}``.


[^Takens1985]: Takens, On the numerical determination of the dimension of an attractor, in: B.H.W. Braaksma, B.L.J.F. Takens (Eds.), Dynamical Systems and Bifurcations, in: Lecture Notes in Mathematics, Springer, Berlin, 1985, pp. 99–106.
[^Theiler1988]: Theiler, [Lacunarity in a best estimator of fractal dimension. Physics Letters A, 133(4–5)](https://doi.org/10.1016/0375-9601(88)91016-X)
[^Shirer1997]: Shirer et al., [Estimating the Correlation Dimension of Atmospheric Time Series. Journal of the atmospheric sciences, 54.1 (1997), pp. 211–230.](https://doi.org/10.1175/1520-0469(1997)054%3C0211:ETCDOA%3E2.0.CO;2)
"""
function shirer_estimate(X, εmax, p, k, metric = Chebyshev(), εmin=0)
    n, nom, den, N = 0, zero(eltype(X)),zero(eltype(X)), length(X)
    @inbounds for i in 1:N
        for j in i+1:N
            d = evaluate(metric, X[i], X[j])
            if εmin < d < εmax
                nom += (d/ε_max)^p * abs(log(d/εmax))^(k-1)
                den += (d/ε_max)^p * abs(log(d/εmax))^k
            end
        end
    end
    return nom/den-p
end
