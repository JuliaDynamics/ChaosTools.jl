#######################################################################################
# Original correlation sum
#######################################################################################
using Distances, Roots
export kernelprob, correlationsum, grassberger, boxed_correlationdim, boxed_correlationsum,
estimate_r0_buenoorovio

"""
    kernelprob(X, ε; norm = Euclidean()) → p::Probabilities
Associate each point in `X` (`Dataset` or timesries) with a probability `p` using the
"kernel estimation" (also called "nearest neighbor kernel estimation" and other names):
```math
p_j = \\frac{1}{N}\\sum_{i=1}^N B(||X_i - X_j|| < \\epsilon)
```
where ``N`` is its length and ``B`` gives 1 if the argument is `true`.

See also [`genentropy`](@ref) and [`correlationsum`](@ref).
`kernelprob` is equivalent with `probabilities(X, NaiveKernel(ε, TreeDistance(norm)))`.
"""
function kernelprob(X, ε; norm = Euclidean())
    probabilities(X, NaiveKernel(ϵ, TreeDistance(norm)))
end


"""
    correlationsum(X, ε::Real; w = 0, norm = Euclidean(), q = 2) → C_q(ε)
Calculate the `q`-order correlation sum of `X` (`Dataset` or timeseries)
for a given radius `ε` and `norm`, using the formula:
```math
C_2(\\epsilon) = \\frac{2}{(N-w)(N-w-1)}\\sum_{i=1}^{N}\\sum_{j=1+w+i}^{N} B(||X_i - X_j|| < \\epsilon)
```
for `q=2` and
```math
C_q(\\epsilon) = \\frac{1}{(N-2w)(N-2w-1)^{(q-1)}} \\sum_{i=1}^N\\left[\\sum_{j:|i-j| > w} B(||X_i - X_j|| < \\epsilon)\\right]^{q-1}
```
for `q≠2`, where ``N`` is its length and ``B`` gives 1 if the argument is
`true`. `w` is the [Theiler window](@ref). If `ε` is a vector its values have to be
ordered. See the book "Nonlinear Time Series Analysis"[^Kantz2003], Ch. 6, for
a discussion around `w` and choosing best values and Ch. 11.3 for the
definition of the q-order correlationsum.

    correlationsum(X, εs::AbstractVector; w, norm, q) → C_q(ε)

If `εs` is a vector, `C_q` is calculated for each `ε ∈ εs`.
If also `q=2`, some strong optimizations are done, but this requires the allocation
a matrix of size `N×N`. If this is larger than your available memory please use instead:
```julia
[correlationsum(..., ε) for ε in εs]
```

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
        for j in i+1+w:N
            C += evaluate(norm, x, X[j]) < ε
        end
    end
    return C * 2 / ((N-w-1)*(N-w))
end

function correlationsum_q(X, ε::Real, q, norm = Euclidean(), w = 0)
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

# Optimized version
function correlationsum_2(X, εs::AbstractVector, norm = Euclidean(), w = 0)
    @assert issorted(εs) "Sorted εs required for optimized version."
    d = distancematrix(X, norm)
    Cs = zeros(eltype(X), length(εs))
    N = length(X)
    factor = 2/((N-w)*(N-1-w))
    # First loop: mid-way ε until lower saturation point (C=0)
    for k in length(εs)÷2:-1:1
        ε = εs[k]
        for i in 1:N
            @inbounds Cs[k] += count(d[j, i] < ε for j in i+1+w:N)
        end
        Cs[k] == 0 && break
    end
    # Second loop: mid-way ε until higher saturation point (C=max)
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
    Nε, T, N = length(εs), eltype(X), length(X)
    Cs = zeros(T, Nε)
    normalisation = (N-2w)*(N-2w-one(T))^(q-1)
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


################################################################################
# Correlationsum, but we distributed data to boxes beforehand
################################################################################
function boxed_correlationdim(data; M = size(data, 2), q = 2)
    r0 = estimate_r0_buenoorovio(data, M)
    ε0 = min_pairwise_distance(data)[2]
    εs = 10 .^ range(log10(ε0), log10(r0), length = 16)
    boxed_correlationdim(data, εs, r0; M = M, q = q)
end

"""
    boxed_correlationdim(data, εs, r0 = maximum(εs); q = 2, M = size(data, 2))
    boxed_correlationdim(data; q = 2, M = size(data, 2))
Estimates the box assisted q-order correlation dimension[^Kantz2003] out of a
dataset `data` for radii `εs` by splitting the data into boxes of size `r0`
beforehand. If `M` is unequal to the dimension of the data, only the first `m`
dimensions are considered. The method of splitting the data into boxes was
implemented according to Theiler[^Theiler1987]. If only a dataset is given the
radii `εs` and boxsize `r0` are chosen by calculating
[`estimate_r0_buenoorovio`](@ref).

This method splits the data into boxes, calculates the q-order correlation sum
C_q(ε) for every `ε ∈ εs` and fits a line through the longest linear looking
region of the curve `(log(εs), log(C_q(εs)))`. The gradient of this line is the
dimension.

The function is explicitly optimized for `q = 2`.

See also: [`correlation_boxing`](@ref),
[`boxed_correlationsum`](@ref) and [`q_order_correlationsum`] (@ref)

[^Kantz]: Kantz, H., & Schreiber, T. (2003). [More about invariant quantities. In Nonlinear Time Series Analysis (pp. 197-233). Cambridge: Cambridge University Press.](https://doi:10.1017/CBO9780511755798.013)

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)
"""
function boxed_correlationdim(data, εs, r0 = maximum(ε); q = 2, M = size(data, 2))
    @assert M ≤ size(data,2) "Prism dimension has to be lower or equal than " *
    "data dimension."
    dd = boxed_correlationsum(data, εs, r0; q = q, M = M)
    linear_region(log.(εs), log.(dd), tol = 0.1)[2]
end

"""
    boxed_correlationsumdata, εs, r0 = maximum(ε); q = 2 , M = size(data, 2))
Distribute `data` into boxes of size `r0`. The `q`-order correlationsum
`C_q(ε)` is then calculated for every `ε ∈ εs` and each of the boxes to then be
summed up afterwards. If `M` is unequal to the dimension of the data, only the
first `m` dimensions are considered for the box distribution.

See also: [`boxed_correlationdim`](@ref)
"""
function boxed_correlationsum(data, εs, r0 = maximum(εs); q = 2, M = size(data, 2))
    @assert M ≤ size(data, 2) "Prism dimension has to be lower or equal than " *
    "data dimension."
    boxes, contents = correlation_boxing(data, r0, M)
    if q == 2
        boxed_correlationsum_2(boxes, contents, data, εs)
    else
        boxed_correlationsum_q(boxes, contents, data, εs, q)
    end
end

"""
    correlation_boxing(data, r0, M = size(data, 2))
Distributes the `data` points into boxes of size `r0`. Returns box positions
and the contents of each box as two separate vectors. Implemented according to
the paper by Theiler[^Theiler1987] improving the algorithm by Grassberger and
Procaccia[^Grassberger1983]. If `M` is smaller than the dimension of the data,
only the first `M` dimensions are considered for the distribution into boxes.

See also: [`estimate_r0_theiler`](@ref), [`estimate_r0_buenoorovio`](@ref),
[`grassberger`](@ref).

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function correlation_boxing(data, r0, M = size(data, 2))
    @assert M ≤ size(data, 2) "Prism dimension has to be lower or equal than" *
    "data dimension."
    mini = minima(data)[1:M]

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point[1:M] - mini)/r0), data)
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
    boxed_correlationsum_2(boxes, contents, data, εs)
For a vector of `boxes` and the indices of their `contents` inside of `data`,
calculate the classic correlationsum of a radius or multiple radii `εs`.
"""
function boxed_correlationsum_2(boxes, contents, data, εs)
    Cs = zeros(eltype(data), length(εs))
    N = length(data)
    for index in 1:length(boxes)
        indices = find_neighborboxes_2(index, boxes, contents)
        X = data[contents[index]]
        Y = data[indices]
        Cs .+= inner_correlationsum_2(X, Y, εs)
    end
    Cs .* (2 / (N * (N - 1)))
end

"""
    find_neighborboxes_2(index, boxes, contents) → indices
For an `index` into `boxes` all neighbouring boxes beginning from the current one are searched. If the found box is indeed a neighbour, the `contents` of that box are added to `indices`.
"""
function find_neighborboxes_2(index, boxes, contents)
    indices = Int[]
    box = boxes[index]
    N_box = length(boxes)
    for index2 in index:N_box
        if evaluate(Chebyshev(), box, boxes[index2]) < 2
            indices = vcat(indices, contents[index2])
        end
    end
    indices
end

"""
    inner_correlationsum_2(X, Y, εs; norm = Euclidean())
Calculates the classic correlation sum for values `X` inside a box, considering `Y` consisting of all values in that box and the ones in neighbouring boxes for all distances `ε ∈ εs` calculated by `norm`.

See also: [`correlationsum`](@ref)
"""
function inner_correlationsum_2(X, Y, εs; norm = Euclidean())
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs, Ny, Nε = zeros(length(εs)), length(Y), length(εs)
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
    return Cs
end

"""
    boxed_correlationsum_q(boxes, contents, data, εs, q)
For a vector of `boxes` and the indices of their `contents` inside of `data`,
calculate the `q`-order correlationsum of a radius or radii `εs`.
"""
function boxed_correlationsum_q(boxes, contents, data, εs, q)
    Cs = zeros(eltype(data), length(εs))
    N = length(data)
    for index in 1:length(boxes)
        indices = find_neighborboxes_q(index, boxes, contents, q)
        X = data[contents[index]]
        Y = data[indices]
        Cs .+= inner_correlationsum_q(X, Y, εs, q)
    end
    Cs ./ (N * (N - 1) ^ (q-1))
end

"""
    find_neighborboxes_q(index, boxes, contents, q) → indices
For an `index` into `boxes` all neighbouring boxes are searched. If the found
box is indeed a neighbour, the `contents` of that box are added to `indices`.
"""
function find_neighborboxes_q(index, boxes, contents, q)
    indices = Int[]
    box = boxes[index]
    for (index2, box2) in enumerate(boxes)
        if evaluate(Chebyshev(), box, box2) < 2
            indices = vcat(indices, contents[index2])
        end
    end
    indices
end

"""
    inner_correlationsum_q(q::Real, X, Y, εs; norm = Euclidean())
Calculates the `q`-order correlation sum for values `X` inside a box, considering `Y` consisting of all values in that box and the ones in neighbouring boxes for all distances `ε ∈ εs` calculated by `norm`.

See also: [`correlationsum`](@ref)
"""
function inner_correlationsum_q(X, Y, εs, q::Real; norm = Euclidean())
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs, Ny, Nε = zeros(length(εs)), length(Y), length(εs)
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
    estimate_r0_buenoorovio(X, M = size(X, 2))
Estimates a reasonable size for boxing the time series `X` proposed by
Bueno-Orovio and Pérez-García[^Bueno2007] before calculating the correlation
dimension as presented by Theiler[^Theiler1983]. If instead of boxes, prisms
are chosen everything stays the same but `M` is the dimension of the prism.
To do so the dimension `ν` is estimated by running the algorithm by Grassberger
and Procaccia[^Grassberger1983] with `√N` points where `N` is the number of
total data points.
An effective size `ℓ` of the attractor is calculated by boxing a small subset
of size `N/10` into boxes of sidelength `r_ℓ` and counting the number of filled
boxes `η_ℓ`.
```math
\\ell = r_\\ell \\eta_\\ell ^{1/\\nu}
```
The optimal number of filled boxes `η_opt` is calculated by minimising the number of calculations.
```math
\\eta_\\textrm{opt} = N^{2/3}\\cdot \\frac{3^\\nu - 1}{3^M - 1}^{1/2}.
```
`M` is the dimension of the data or the number of edges on the prism that don't
span the whole dataset.

Then the optimal boxsize ``r_0`` computes as
```math
r_0 = \\ell / \\eta_\\textrm{opt}^{1/\\nu}.
```

[^Bueno2007]: Bueno-Orovio and Pérez-García, [Enhanced box and prism assisted algorithms for computing the correlation dimension. Chaos Solitons & Fractrals, 34(5)](https://doi.org/10.1016/j.chaos.2006.03.043)

[^Theiler1987]: Theiler, [Efficient algorithm for estimating the correlation dimension from a set of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]: Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)](https://journals-aps-org.e-bis.mpimet.mpg.de/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function estimate_r0_buenoorovio(X, M = size(X, 2))
    mini, maxi = minmaxima(X)
    N = length(X)
    R = maximum(maxi .- mini)
    # The possibility of a bad pick exists, if so, the calculation is repeated.
    ν = zero(eltype(X))
    # Sample N/10 datapoints out of data for rough estimate of effective size.
    sample1 = X[unique(rand(1:N, N÷10))] |> Dataset
    r_ℓ = R / 10
    η_ℓ = length(correlation_boxing(sample1, r_ℓ)[1])
    r0 = zero(eltype(X))
    while true
        # Sample √N datapoints for rough dimension estimate
        sample2 = X[unique(rand(1:N, ceil(Int, sqrt(N))))] |> Dataset
        # Define logarithmic series of radii.
        lower = log10(min_pairwise_distance(X)[2])
        εs = 10 .^ range(lower, stop = log10(R), length = 16)
        # Estimate ν from a sample using the Grassberger Procaccia algorithm.
        cm = correlationsum(sample2, εs)
        ν = linear_region(log.(εs), log.(cm), tol = 0.5)[2]
        # Estimate the effictive size of the chaotic attractor.
        ℓ = r_ℓ * η_ℓ^(1/ν)
        # Calculate the optimal number of filled boxes according to Bueno-Orovio
        η_opt = N^(2/3) * ((3^ν - 1/2) / (3^M - 1))^(1/2)
        # The optimal box size is the effictive size divided by the box number # to the power of the inverse dimension.
        r0 = ℓ / η_opt^(1/ν)
        !isnan(r0) && break
    end
    return r0
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
