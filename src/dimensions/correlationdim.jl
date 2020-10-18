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
    correlationsum(X, ε; w = 0, norm = Euclidean(), q = 2) → C_q(ε)
Calculate the `q`-order correlation sum of `X` (`Dataset` or timeseries)
for a given radius `ε` and `norm`, using the formula:
```math
C_2(\\epsilon) = \\frac{2}{(N-w)(N-w-1)}\\sum_{i=1}^{N}\\sum_{j=1+w+i}^{N} I(||X_i - X_j|| < \\epsilon)
```
for `q=2` and
```math
C_q(\\epsilon) = \\frac{1}{(N-w)(N-w-1)^{(q-1)}} \\sum_{i=1}^N\\left[\\sum_{|i-j| > w} I(||X_i - X_j|| < \\epsilon)\\right]^{q-1}
```
for `q≠2`, where ``N`` is its length and ``I`` gives 1 if the argument is `true`. `w` is the Theiler window, a correction to the correlation sum that skips points
that are temporally close with each other, with the aim of removing spurious correlations.
See the book "Nonlinear Time Series Analysis"[^Kantz2003], Ch. 6, for a discussion
around `w` and choosing best values and Ch. 11.3 for the definition of the q-order correlationsum.

See [`grassberger`](@ref) for more.
See also [`takens_best_estimate`](@ref).

[^Kantz]: Kantz, H., & Schreiber, T. (2003). [More about invariant quantities. In Nonlinear Time Series Analysis (pp. 197-233). Cambridge: Cambridge University Press.](https://doi:10.1017/CBO9780511755798.013)
"""
function correlationsum(X, ε::Real; q = 2, norm = Euclidean(), w = 0)
    N, C = length(X), 0.
    if q == 2
        for (i, x) in enumerate(X)
            # assumes that the first Nx elements are X itself
            for j in i+1+w:N
                C += evaluate(norm, x, X[j]) < ε
            end
        end
        return C * 2 / ((N-w-1)*(N-w))
    else
        for i in 1+w:N-w
            x = X[i]
            C_current = 0.
            # computes all distances from 0 up to i-w
            for j in 1:i-w-1
                C_current += evaluate(norm, x, X[j]) < ε
            end
            # computes all distances after i+w till the end
            for j in i+w+1:N
                C_current += evaluate(norm, x, X[j]) < ε
            end
            C += C_current^(q - 1)
        end
        return C / ((N-2w)*(N-2w-1)^(q-1))
    end
end



"""
correlationsum(X, εs::AbstractVector; q = 2, norm = Euclidean(), w = 0) → [C_q(ε) for ε ∈ εs]
If `εs` is a vector containing radii, a slight optimisation is applied to prevent calculating distances twice. For this optimisation `εs` needs to be of increasing order.
"""
function correlationsum(X, εs::AbstractVector; q = 2, norm = Euclidean(), w = 0)
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs, N, Nε = zeros(length(εs)), length(X), length(εs)
    if q == 2
        for (i, x) in enumerate(X)
            for j in i+1+w:N
                dist = evaluate(norm, X[j], x)
                for k in Nε:-1:1
                    if dist < εs[k]
                        Cs[k] += 1
                    else
                        break
                    end
                end
            end
        end
        return Cs .* 2 / ((N-w)*(N-w-1))
    else
        for i in 1+w:N-w
            x = X[i]
            C_current = zeros(Nε)
            # Compute distances from 1 to the start of the w-intervall around i.
            for j in 1:i-w-1
                dist = evaluate(norm, x, X[j])
                for k in Nε:-1:1
                    if dist < εs[k]
                        C_current[k] += 1
                    else
                        break
                    end
                end
            end
            # Compute distances from the end of w-intervall around i till the end.
            for j in i+w+1:N
                dist = evaluate(norm, x, X[j])
                for k in Nε:-1:1
                    if dist < εs[k]
                        C_current[k] += 1
                    else
                        break
                    end
                end
            end
            Cs .+= C_current.^(q-1)
        end
        return Cs ./ ((N-2w)*(N-2w-1)^(q-1))
    end
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
    takens_best_estimate(X, εmax, metric = Chebyshev()) → D_C
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
D_C \\approx - \\frac{1}{\\eta},\\quad \\eta = \\frac{1}{N^*}\\sum_{[i, j]^*}\\log(||X_i - X_j|| / \\epsilon_\\text{max})
```
where the sum happens for all ``i, j`` so that ``i < j`` and ``||X_i - X_j|| < \\epsilon_\\text{max}``.

If `X` comes from a delay coordinates embedding of a timseries `x`, a recommended value
for ``\\epsilon_\\text{max}`` is `std(x)/4`.

[^Takens1985]: Takens, On the numerical determination of the dimension of an attractor, in: B.H.W. Braaksma, B.L.J.F. Takens (Eds.), Dynamical Systems and Bifurcations, in: Lecture Notes in Mathematics, Springer, Berlin, 1985, pp. 99–106.
[^Theiler1988]: Theiler, [Lacunarity in a best estimator of fractal dimension. Physics Letters A, 133(4–5)](https://doi.org/10.1016/0375-9601(88)91016-X)
"""
function takens_best_estimate(X, εmax, metric = Chebyshev())
    n, η, N = 0, zero(eltype(X)), length(X)
    @inbounds for i in 1:N
        for j in i+1:N
            d = evaluate(metric, X[i], X[j])
            if d < εmax
                n += 1
                η += log(d/εmax)
            end
        end
    end
    return -n/η
end
