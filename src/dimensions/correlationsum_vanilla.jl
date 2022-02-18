import ProgressMeter
#######################################################################################
# Original correlation sum
#######################################################################################
using Distances, Roots
export correlationsum, grassberger_dim, boxed_correlationsum,
estimate_r0_buenoorovio, data_boxing, autoprismdim, estimate_r0_theiler

"""
    correlationsum(X, ε; w = 0, norm = Euclidean(), q = 2) → C_q(ε)
Calculate the `q`-order correlation sum of `X` (`Dataset` or timeseries)
for a given radius `ε` and `norm`. They keyword `show_progress = true` can be used
to display a progress bar for large `X`.

    correlationsum(X, εs::AbstractVector; w, norm, q) → C_q(ε)

If `εs` is a vector, `C_q` is calculated for each `ε ∈ εs` more efficiently.
If also `q=2`, we attempt to do further optimizations, if the allocation
a matrix of size `N×N` is possible.

The function [`boxed_correlationsum`](@ref) is faster and should be preferred over this one.

## Description
The correlation sum is defined as follows for `q=2`:
```math
C_2(\\epsilon) = \\frac{2}{(N-w)(N-w-1)}\\sum_{i=1}^{N}\\sum_{j=1+w+i}^{N} 
B(||X_i - X_j|| < \\epsilon)
```
for as follows for `q≠2`
```math
C_q(\\epsilon) = \\left[\\frac{1}{\\alpha} \\sum_{i=w+1}^{N-w}
\\left[\\sum_{j:|i-j| > w} B(||X_i - X_j|| < \\epsilon)\\right]^{q-1}\\right]^{1/(q-1)}
```
where
```math
\\alpha = (N-2w)(N-2w-1)^{(q-1)}
```
with ``N`` the length of `X` and ``B`` gives 1 if its argument is
`true`. `w` is the [Theiler window](@ref). 
See the article of Grassberger for the general definition [^Grassberger2007] and 
the book "Nonlinear Time Series Analysis" [^Kantz2003], Ch. 6, for
a discussion around choosing best values for `w`, and Ch. 11.3 for the
explicit definition of the q-order correlationsum.

The scaling of ``\\log C_q`` versus ``\\log \\epsilon`` approximates the q-order
generalized (Rényi) dimension.

[^Grassberger2007]: 
    Peter Grassberger (2007) [Grassberger-Procaccia algorithm. Scholarpedia, 
    2(5):3043.](http://dx.doi.org/10.4249/scholarpedia.3043)

[^Kantz2003]: 
    Kantz, H., & Schreiber, T. (2003). [Nonlinear Time Series Analysis, 
    Cambridge University Press.](https://doi.org/10.1017/CBO9780511755798)
"""
function correlationsum(X, ε; q = 2, norm = Euclidean(), w = 0, show_progress = false)
    q ≤ 1 && @warn "The correlation sum is ill-defined for q ≤ 1."
    if q == 2
        correlationsum_2(X, ε, norm, w, show_progress)
    else
        correlationsum_q(X, ε, q, norm, w, show_progress)
    end
end

function correlationsum_2(X, ε::Real, norm, w, show_progress)
    N = length(X)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Correlation sum: ", dt = 1.0)
    end
    C = zero(eltype(X))
    @inbounds for (i, x) in enumerate(X)
        for j in i+1+w:N
            C += evaluate(norm, x, X[j]) < ε
        end
        show_progress && ProgressMeter.next!(progress)
    end
    return C * 2 / ((N-w-1)*(N-w))
end

function correlationsum_q(X, ε::Real, q, norm, w, show_progress)
    N, C = length(X), zero(eltype(X))
    normalisation = (N-2w)*(N-2w-one(eltype(X)))^(q-1)
    if show_progress
        progress = ProgressMeter.Progress(length(1+w:N-w); desc="Correlation sum: ", dt=1)
    end
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
        show_progress && ProgressMeter.next!(progress)
    end
    return (C / normalisation) ^ (1 / (q-1))
end

# Optimized version
function correlationsum_2(X, εs::AbstractVector, norm, w, show_progress)
    @assert issorted(εs) "Sorted `ε` required for optimized version."
    d = try
        distancematrix(X, norm)
    catch err
        @warn "Couldn't create distance matrix ($(typeof(err))). Using slower algorithm..."
        return [correlationsum_2(X, ε, norm, w, show_progress) for ε in εs]
    end
    return correlationsum_2_fb(X, εs, d, w, show_progress) # function barrier
end
function correlationsum_2_fb(X, εs, d, w, show_progress)
    Cs = zeros(eltype(X), length(εs))
    N = length(X)
    factor = 2/((N-w)*(N-1-w))
    if show_progress
        K = length(length(εs)÷2:-1:1)
        M = K + length((length(εs)÷2 + 1):length(εs))
        progress = ProgressMeter.Progress(M; desc = "Correlation sum: ", dt = 1.0)
    end

    # First loop: mid-way ε until lower saturation point (C=0)
    for (ki, k) in enumerate(length(εs)÷2:-1:1)
        ε = εs[k]
        for i in 1:N
            @inbounds Cs[k] += count(d[j, i] < ε for j in i+1+w:N)
        end
        show_progress && ProgressMeter.update!(progress, ki)
        Cs[k] == 0 && break
    end
    # Second loop: mid-way ε until higher saturation point (C=max)
    for (ki, k) in enumerate((length(εs)÷2 + 1):length(εs))
        ε = εs[k]
        for i in 1:N
            @inbounds Cs[k] += count(d[j, i] < ε for j in i+1+w:N)
        end
        show_progress && ProgressMeter.update!(progress, ki+K)
        if Cs[k] ≈ 1/factor
            Cs[k:end] .= 1/factor
            break
        end
    end
    show_progress && ProgressMeter.finish!(progress)
    return Cs .* factor
end

function correlationsum_q(X, εs::AbstractVector, q, norm, w, show_progress)
    @assert issorted(εs) "Sorted εs required for optimized version."
    Nε, T, N = length(εs), eltype(X), length(X)
    Cs = zeros(T, Nε)
    normalisation = (N-2w)*(N-2w-one(T))^(q-1)
    if show_progress
        progress = ProgressMeter.Progress(length(1+w:N-w); desc="Correlation sum: ", dt=1)
    end
    for i in 1+w:N-w
        x = X[i]
        C_current = zeros(T, Nε)
        # Compute distances for j outside the Theiler window
        for j in Iterators.flatten((1:i-w-1, i+w+1:N))
            dist = evaluate(norm, x, X[j])
            for k in Nε:-1:1
                if dist < εs[k]
                    C_current[k] += 1
                else
                    break
                end
            end
        end
        Cs .+= C_current .^ (q-1)
        show_progress && ProgressMeter.next!(progress)
    end
    return (Cs ./ normalisation) .^ (1/(q-1))
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
    grassberger_dim(data, εs = estimate_boxsizes(data); kwargs...) → D_C
Use the method of Grassberger and Proccacia[^Grassberger1983], and the correction by
Theiler[^Theiler1986], to estimate the correlation dimension `D_C` of the given `data`.

This function does something extremely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
return linear_region(log.(sizes), log(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus log(ε).
See [`generalized_dim`](@ref) for a more thorough explanation.

See also [`takens_best_estimate`](@ref).

[^Grassberger1983]: 
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)

[^Theiler1986]: 
    Theiler, [Spurious dimension from correlation algorithms applied to limited time-series
    data. Physical Review A, 34](https://doi.org/10.1103/PhysRevA.34.2427)
"""
function grassberger_dim(data::AbstractDataset, εs = estimate_boxsizes(data); kwargs...)
    @warn "`grassberger_dim` is deprecated and will be removed in future versions."
    cm = correlationsum(data, εs; kwargs...)
    return linear_region(log.(εs), log.(cm))[2]
end
