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

################################################################################
# q-order correlationsum
################################################################################
"""
	q_order_correlationsum(q, X, ε, norm = Euclidean()) → C_q(ε)
	q_order_correlationsum(q, X, Y, ε, norm = Euclidean()) → C_q(ε)
Calculates the q-order correlation sum for data points `X`. `q` is the exponent of all points that are in radius `ε` of a point. Uses the method by Kantz and Schreiber[^Kantz] In accordance to the formula:
```math
C_q(\\varepsilon) = \\frac{1}{N(N-1)^{(q-1)}} \\sum_{i=1}^N\\left[\\sum_{i \\ne j} \\Theta(\\varepsilon - ||X_i - X_j||)\\right]^{q-1}
```
where ``\\Theta`` is the Heaviside function yielding one if the argument is greater than 1.

If `ε` is a vector of radii, it uses a slight optimisation to calculate the correlationsums. In this case `ε` should be ordered in increasing order.

If `X` and `Y` are given, it computes the between all points in `X` and `Y`. In the folllowing formula:
```math
C_q(\\varepsilon) = \\sum_{i=1}^N\\left[\\sum_{i \\ne j}^M \\Theta(\\varepsilon - ||X_i - Y_j||)\\right]^{q-1}.
```
Here ``M`` is the number of points in `Y`. Since the version with `X` and `Y` is optimized for data that was boxed beforehand, the normalisation is not calculated. Note that the first elements of `Y` should be `X` for the case of `q = 2` to allow for optimisation.

[^Kantz]: Kantz, H., & Schreiber, T. (2003). [More about invariant quantities. In Nonlinear Time Series Analysis (pp. 197-233). Cambridge: Cambridge University Press.](https://doi:10.1017/CBO9780511755798.013)
"""
function q_order_correlationsum(q, X, ε, norm = Euclidean())
	N = length(X)
	C_q = q_order_correlationsum(q, X, X, ε, norm) ./ (N * (N-1)^(q-1))
end

function q_order_correlationsum(q, X, Y, ε::Real, norm = Euclidean())
	C_q = 0.
	if q == 2
		Ny = length(Y)
		for (ix, x) in enumerate(X)
			# assumes that in case of q == 2 the first Nx elements are X itself
			for iy in ix+1:Ny
				y = Y[iy]
				C_q += evaluate(norm, x, y) < ε
			end
		end
		# corrects the omitted terms
		C_q *= 2
	else
		for x in X
			C_current
			for y in Y
				C_current += evaluate(norm, x, y) < ε
			end
			# the minus 1 is for correction of i = j,
			# which is cheaper than calculating [1:i-1;i+1:N].
			C_q += (C_current - 1)^(q - 1)
		end
	end
	C_q
end

function q_order_correlationsum(q, X, Y, εs::AbstractVector, norm = Euclidean())
	@assert issorted(εs) "Sorted εs required for optimized version."
	Nε = length(εs)
	C_qs = zeros(Nε)
	if q == 2
		Ny = length(Y)
		for (ix, x) in enumerate(X)
			for iy in ix+1:Ny
				dist = evaluate(norm, Y[iy], x)
				for iε in Nε:-1:1
					if dist < εs[iε]
						C_qs[iε] += 1
					else
						break
					end
				end
			end
		end
		C_qs .*= 2
	else
		for x in X
			C_current = zeros(Nε)
			for y in Y
				dist = evaluate(norm, x, y)
				for iε in Nε:-1:1
					if dist < εs[iε]
						C_current[iε] += 1
					else
						break
					end
				end
			end
			# The minus 1 corrects i = j.
			C_qs .+= (C_current .- 1).^(q-1)
		end
	end
	C_qs
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
