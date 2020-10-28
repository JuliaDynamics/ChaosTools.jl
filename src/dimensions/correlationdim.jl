#######################################################################################
# Original correlation sum
#######################################################################################
using Distances, Roots
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
that are temporally close with each other, with the aim of removing spurious correlations. If `ε` is a vector its values have to be ordered. 
See the book "Nonlinear Time Series Analysis"[^Kantz2003], Ch. 6, for a discussion
around `w` and choosing best values and Ch. 11.3 for the definition of the q-order correlationsum.

See [`grassberger`](@ref) for more.
See also [`takens_best_estimate`](@ref).

[^Kantz]: Kantz, H., & Schreiber, T. (2003). [More about invariant quantities. In Nonlinear Time Series Analysis (pp. 197-233). Cambridge: Cambridge University Press.](https://doi:10.1017/CBO9780511755798.013)
"""
function correlationsum(X, ε; q = 2, norm = Euclidean(), w = 0)
    if q == 2
        correlationsum_2(X, ε, norm, w)
    else
        correlationsum_q(X, ε, q, norm, w)
    end
end

function correlationsum_2(X, ε::Real, norm = Euclidean(), w = 0)
    N, C = length(X), zero(eltype(X))
    for (i, x) in enumerate(X)
        # assumes that the first Nx elements are X itself
        for j in i+1+w:N
            C += evaluate(norm, x, X[j]) < ε
        end
    end
    return C * 2 / ((N-w-1)*(N-w))
end

function correlationsum_q(X, ε::Real, q, norm = Euclidean(), w = 0)
    q <= 1 && @warn "This function is currently not specialized for q <= 1" *
    " and may show unexpected behaviour for these values."
    N, C = length(X), zero(eltype(X))
    normalisation = (N-2w)*(N-2w-one(eltype(X)))^(q-1)
    for i in 1+w:N-w
        x = X[i]
        C_current = zero(eltype(X))
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
    return C / normalisation
end


function correlationsum_2(X, εs::AbstractVector, norm = Euclidean(), w = 0)
    @assert issorted(εs) "Sorted εs required for optimized version."
    d = distancematrix(X, norm)
    Cs = zeros(eltype(X), length(εs))
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

function correlationsum_q(X, εs::AbstractVector, q, norm = Euclidean(), w = 0)
    @assert issorted(εs) "Sorted εs required for optimized version."
    q <= 1 && @warn "This function is currently not specialized for q <= 1" *
    " and may show unexpected behaviour for these values."
    Nε, T, N = length(εs), eltype(X), length(X)
    Cs = zeros(T, Nε)
    normalisation = (N-2w)*(N-2w-one(T))^(q-1)
    for i in 1+w:N-w
        x = X[i]
        C_current = zeros(T, Nε)
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
    return Cs ./ normalisation
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
    takens_best_estimate(X, εmax, metric = Chebyshev(),εmin = 0) → D_C, D_C_95u, D_C_95l
Use the so-called "Takens' best estimate" [^Takens1985][^Theiler1988]
method for estimating the correlation dimension
`D_C` and the upper (`D_C_95u`) and lower (`D_C_95l`) confidence limit for the given dataset `X`.

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

The confidence limits are estimated from the log-likelihood function by finding
the values of `D_C` where the function has fallen by 2 from its maximum, see e.g.
[^Barlow] chapter 5.3
Because the CLT does not apply (no independent measurements), the limits are not
neccesarily symmetric.

According to [^Borovkova1999], introducing a lower cutoff `εmin` can make the
algorithm more stable (no divergence), this option is given but defaults to zero.

If `X` comes from a delay coordinates embedding of a timseries `x`, a recommended value
for ``\\epsilon_\\text{max}`` is `std(x)/4`.

[^Takens1985]: Takens, On the numerical determination of the dimension of an attractor, in: B.H.W. Braaksma, B.L.J.F. Takens (Eds.), Dynamical Systems and Bifurcations, in: Lecture Notes in Mathematics, Springer, Berlin, 1985, pp. 99–106.
[^Theiler1988]: Theiler, [Lacunarity in a best estimator of fractal dimension. Physics Letters A, 133(4–5)](https://doi.org/10.1016/0375-9601(88)91016-X)
[^Borovkova1999]: Borovkova et al., [Consistency of the Takens estimator for the correlation dimension. The Annals of Applied Probability, 9, 05 1999.](https://doi.org/10.1214/aoap/1029962747)
[^Barlow]: Barlow, R., Statistics - A Guide to the Use of Statistical Methods in the Physical Sciences. Vol 29. John Wiley & Sons, 1993
"""
function takens_best_estimate(X, εmax, metric = Chebyshev(); εmin=0)
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
    # bias-corrected version (log-likelihood function shifted on x-axis)
    α = -(n-1)/η
    # biased version (maximum of original log-likelihood function)
    α_b = -n/η
    # value of maximum of original log-likelihood function
    mxl = n*log(α_b) + α_b * η
    # at the 95%-confidence interval, the log-l function has dropped by 2
    # -> log_l(x) - mxl + 2 = 0
    # this is a result of the invariance of the MLE, a really nice property
    # these limits are not going to be perfectly symmetric (CLT does not apply)
    mn, mx = fzeros(x-> n * log(x) + η * x - mxl +2 , 0,2*α)

    # Since the bias-correction is just a shift of the log-l function on the
    # x-axis, we can easily shift the confidence limits by the bias α-α_b
    α95u = α - α_b + mn
    α95l = α - α_b + mx

    return α, α95u, α95l
end
