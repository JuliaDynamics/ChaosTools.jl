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
# Correlationsum, but we distributed data to boxes beforehand
################################################################################
"""
    boxed_correlationdim(data, r0, εs, q = 2)
Estimates the q-order correlation dimension[^Kantz2003] out of a dataset `data` for radii `εs` by splitting the data into boxes of size `r0` beforehand. The method of splitting the data into boxes was mostly copied by the method of Theiler[^Theiler1987].

This method splits the data into boxes, calculates the q-order correlation sum C_q(ε) and fits a line through the longest linear looking region of the curve `(log(εs), log(C_q(εs)))`. The gradient of this line is the dimension.

The function is explicitly optimized for `q = 2`.

See also: [`bueno_orovio_correlationdim`](@ref), [`theiler_correlationdim`](@ref), [`correlation_boxing`](@ref), [`boxed_correlationsum`](@ref) and [`q_order_correlationsum`] (@ref)

[^Kantz]: Kantz, H., & Schreiber, T. (2003). [More about invariant quantities. In Nonlinear Time Series Analysis (pp. 197-233). Cambridge: Cambridge University Press.](https://doi:10.1017/CBO9780511755798.013)

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)
"""
function boxed_correlationdim(data, r0, εs, q = 2)
    boxes, contents = correlation_boxing(data, r0)
    dd = boxed_correlationsum(boxes, contents, data, εs, q)
    linear_region(log.(εs), log.(dd), tol = 0.1)[2]
end

"""
    correlation_boxing(data, r0)
Distributes the `data` points into boxes of size `r0`. Returns box positions and the contents of each box as two separate vectors. Implemented according to the paper by Theiler[^Theiler1987] improving the algorithm by Grassberger and Procaccia[^Grassberger1983].

See also: [`estimate_r0_theiler`](@ref), [`estimate_r0_buenoorovio`](@ref), [`grassberger`](@ref).

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function correlation_boxing(data, r0)
    mini = minima(data)

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point - mini)/r0), data)
    permutations = sortperm(bins, alg=QuickSort)

    boxes = unique(bins[permutations])
    contents = Vector{Vector{Int}}()
    sizehint!(contents, length(boxes))

    prior, prior_perm = 1, permutations[1]
    # distributes all permutation indices into boxes
    for (index, perm) in enumerate(permutations)
        if bins[perm] != bins[prior_perm]
            push!(contents, permutations[prior:index-1])
            prior, prior_perm = index, perm
        end
    end
    push!(contents, permutations[prior:end])

    Dataset(boxes), contents
end

"""
    boxed_correlationsum(boxes, contents, data, ε, q = 2)
For a vector of `boxes` and the indices of their `contents` inside of `data`, this function finds the q-order correlationsum for a radius or radii `ε`.

See also: [`boxed_correlationdim`](@ref)
"""
function boxed_correlationsum(boxes, contents, data, ε, q = 2)
    q <= 1 && @warn "This function is currently not specialized for q <= 1" *
    " and may show unexpected behaviour for these values."
    Cs = zeros(Float64, length(ε))
    N = length(data)
    for index in 1:length(boxes)
        indices = find_neighbourboxes(index, boxes, contents, q)
        X = data[contents[index]]
        Y = data[indices]
        Cs .+= correlationsum_boxes(q, X, Y, ε)
    end
    Cs ./ (N * (N - 1) ^ (q-1))
end

"""
    find_neighbourboxes(index, boxes, contents, q) → indices
For an `index` into `boxes` all neighbouring boxes are searched. If the found box is indeed a neighbour, the `contents` of that box are added to `indices`. If for the q-order correlation to be calculated `q = 2`, only those boxes are searched whose index is greater or equal to the original, since the boxes are expected to be ordered and each distance shall only be calculated once.
"""
function find_neighbourboxes(index, boxes, contents, q)
    indices = Int[]
    box = boxes[index]
    if q == 2
        N_box = length(boxes)
        for index2 in index:N_box
            if evaluate(Chebyshev(), box, boxes[index2]) < 2
                indices = vcat(indices, contents[index2])
            end
        end
    else
        for (index2, box2) in enumerate(boxes)
            if evaluate(Chebyshev(), box, box2) < 2
                indices = vcat(indices, contents[index2])
            end
        end
    end
    indices
end

"""
    correlationsum_boxes(q::Real, X, Y, εs; norm = Euclidean())
Calculates the `q`-order correlation sum for values `X` inside a box, considering `Y` consisting of all values in that box and the ones in neighbouring boxes for all distances `ε ∈ εs` calculated by `norm`.

See also: [`correlationsum`](@ref)
"""
function correlationsum_boxes(q::Real, X, Y, εs; norm = Euclidean())
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs, Ny, Nε = zeros(length(εs)), length(Y), length(εs)
    if q == 2
        for (i, x) in enumerate(X)
            for j in i+1:Ny
                dist = evaluate(norm, Y[j], x)
                for k in Nε:-1:1
                    if dist < εs[k]
                        Cs[k] += 1
                    else
                        break
                    end
                end
            end
        end
        Cs .*= 2
    else
        for (i, x) in enumerate(X)
            # accounts for i = j
            C_current = -1 .* ones(Nε)
            for j in 1:Ny
                dist = evaluate(norm, x, Y[j])
                for k in Nε:-1:1
                    if dist < εs[k]
                        C_current[k] += 1
                    else
                        break
                    end
                end
            end
            Cs .+= C_current .^ (q-1)
        end
    end
    return Cs
end

"""
	estimate_r0_theiler(data)
Estimates a reasonable size for boxing the data before calculating the correlation dimension proposed by Theiler[^Theiler1987].
To do so the dimension is estimated by running the algorithm by Grassberger and Procaccia[^Grassberger1983] with `√N` points where `N` is the number of total data points. Then the optimal boxsize ``r_0`` computes as
```math
r_0 = R (2/N)^{1/\\nu}
```
where ``R`` is the size of the chaotic attractor and ``\\nu`` is the estimated dimension.

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function estimate_r0_theiler(data)
	N = length(data)
	mini, maxi = minmaxima(data)
	R = maximum(maxi .- mini)
	# Sample √N datapoints for a rough estimate of the dimension.
	data_sample = data[unique(rand(1:N, ceil(Int, sqrt(N))))] |> Dataset
	# Define radii for the rough dimension estimate
	lower = log10(min_pairwise_distance(data_sample)[2])
	εs = 10 .^ range(lower, stop = log10(R), length = 12)
	# Actually estimate the dimension.
	cm = correlationsum(data_sample, εs)
	ν = linear_region(log.(εs), log.(cm), tol = 0.5)[2]
	# The combination yields the optimal box size
	r0 = R * (2/N)^(1/ν)
end

"""
	estimate_r0_buenoorovio(data)
Estimates a reasonable size for boxing the data proposed by Bueno and Orovio[^Bueno2007] before calculating the correlation dimension as presented by Theiler[^Theiler1983].
To do so the dimension `ν` is estimated by running the algorithm by Grassberger and Procaccia[^Grassberger1983] with `√N` points where `N` is the number of total data points.
An effective size `ℓ` of the attractor is calculated by boxing a small subset of size `N/10` into boxes of sidelength `r_ℓ` and counting the number of filled boxes `η_ℓ`.
```math
\\ell = r_\\ell \\eta_\\ell ^{1/\\nu}
```
The optimal number of filled boxes `η_opt` is calculated by minimising the number of calculations.
```math
\\eta_\\textrm{opt} = N^{2/3}\\cdot \\frac{3^\\nu - 1}{3^m - 1}^{1/2}.
```
`m` is the dimension of the data.
Then the optimal boxsize ``r_0`` computes as
```math
r_0 = \\ell / \\eta_\\textrm{opt}^{1/\\nu}.
```

[^Bueno2007]: Bueno-Orovio and Pérez-García, [Enhanced box and prism assisted algorithms for computing the correlation dimension. Chaos Solitons & Fractrals, 34(5)](https://doi.org/10.1016/j.chaos.2006.03.043)

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function estimate_r0_buenoorovio(data)
    mini, maxi = minmaxima(data)
    N = length(data)
    m = DelayEmbeddings.dimension(data)
	R = maximum(maxi .- mini)
	# Sample √N and N/10 datapoints out of data for rough estimates.
	data_sample1 = data[unique(rand(1:N, ceil(Int, sqrt(N))))] |> Dataset
	data_sample2 = data[unique(rand(1:N, N÷10))] |> Dataset
	# Define logarithmic shaped vector of radii.
	lower = log10(min_pairwise_distance(data_sample1)[2])
	εs = 10 .^ range(lower, stop = log10(R), length = 16)
	# Estimating ν out of a small sample using the Grassberger Procaccia algorithm.
	cm = correlationsum(data_sample1, εs)
	ν = linear_region(log.(εs), log.(cm), tol = 0.5)[2]
	# Calculate the optimal number of filled boxes according to Bueno and Orovio.
	η_opt = N^(2/3) * ((3^ν - 1/2) / (3^m - 1))^(1/2)
	r_ℓ = R / 10
	η_ℓ = length(correlation_boxing(data_sample2, r_ℓ)[1])
	# Estimate the effictive size of the chaotic attractor.
	ℓ = r_ℓ * η_ℓ^(1/ν)
	# The optimal box size is the effictive size divided by the box number to the
	# power of the inverse dimension.
    r0 = ℓ / η_opt^(1/ν)
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
