#=
this file includes functionality for estimating linear scaling regions and
defines the `generalized_dim` function.
=#

export linear_region, linear_regions, estimate_boxsizes, linreg
#####################################################################################
# Functions and methods to deduce linear scaling regions
#####################################################################################
using Statistics
using Statistics: covm, varm
# The following function comes from a version in StatsBase that is now deleted
# StatsBase is copyrighted under the MIT License with
# Copyright (c) 2012-2016: Dahua Lin, Simon Byrne, Andreas Noack, Douglas Bates,
# John Myles White, Simon Kornblith, and other contributors.
"""
    linreg(x, y) -> a, b
Perform a linear regression to find the best coefficients so that the curve:
`y = a + b*x` has the least squared error.
"""
function linreg(x::AbstractVector, y::AbstractVector)
    # Least squares given
    # Y = a + b*X
    # where
    # b = cov(X, Y)/var(X)
    # a = mean(Y) - b*mean(X)
    if size(x) != size(y)
        throw(DimensionMismatch("x has size $(size(x)) and y has size $(size(y)), " *
            "but these must be the same size"))
    end
    mx = Statistics.mean(x)
    my = Statistics.mean(y)
    # don't need to worry about the scaling (n vs n - 1)
    # since they cancel in the ratio
    b = covm(x, mx, y, my)/varm(x, mx)
    a = my - b*mx
    return a, b
end

slope(x, y) = linreg(x, y)[2]


"""
    linear_regions(x, y; dxi::Int = 1, tol = 0.2) -> (lrs, tangents)
Identify regions where the curve `y(x)` is linear, by scanning the
`x`-axis every `dxi` indices sequentially
(e.g. at `x[1] to x[5], x[5] to x[10], x[10] to x[15]` and so on if `dxi=5`).

If the slope (calculated via linear regression) of a region of width `dxi` is
approximatelly equal to that of the previous region,
within tolerance `tol`,
then these two regions belong to the same linear region.

Return the indices of `x` that correspond to linear regions, `lrs`,
and the _correct_ `tangents` at each region
(obtained via a second linear regression at each accumulated region).
"""
function linear_regions(
        x::AbstractVector, y::AbstractVector;
        method = :sequential, dxi::Int = method == :overlap ? 3 : 1, tol = 0.2,
    )
    return if method == :overlap
        linear_regions_overlap(x, y, dxi, tol)
    elseif method == :sequential
        linear_regions_sequential(x, y, dxi, tol)
    end
end

function linear_regions_sequential(x, y, dxi, tol)
    maxit = length(x) ÷ dxi

    tangents = Float64[slope(view(x, 1:max(dxi, 2)), view(y, 1:max(dxi, 2)))]

    prevtang = tangents[1]
    lrs = Int[1] #start of first linear region is always 1
    lastk = 1

    # Start loop over all partitions of `x` into `dxi` intervals:
    for k in 1:maxit-1
        tang = slope(view(x, k*dxi:(k+1)*dxi), view(y, k*dxi:(k+1)*dxi))
        if isapprox(tang, prevtang, rtol=tol, atol = 0)
            # Tanget is similar with initial previous one (based on tolerance)
            continue
        else
            # Tangent is not similar.
            # Push new tangent for a new linear region
            push!(tangents, tang)

            # Set the START of a new linear region
            # which is also the END of the previous linear region
            push!(lrs, k*dxi)
            lastk = k
        end

        # Set new previous tangent (only if it was not the same as current)
        prevtang = tang
    end
    push!(lrs, length(x))
    # create new tangents that do have linear regression weighted
    tangents = Float64[]
    for i in 1:length(lrs)-1
        push!(tangents, linreg(view(x, lrs[i]:lrs[i+1]), view(y ,lrs[i]:lrs[i+1]))[2])
    end
    return lrs, tangents
end

"""
    linear_region(x, y; dxi::Int = 1, tol = 0.2) -> ((ind1, ind2), slope)
Call [`linear_regions`](@ref) and identify and return the largest linear region
and its slope. The region starts and stops at `x[ind1:ind2]`.
"""
function linear_region(x::AbstractVector, y::AbstractVector;
    dxi::Int = 1, tol::Real = 0.2)
    lrs, tangents = linear_regions(x,y; dxi, tol)
    # Find biggest linear region:
    j = findmax(diff(lrs))[2]
    return (lrs[j], lrs[j+1]), tangents[j]
end

#####################################################################################
# Autotomatic estimation for proper `ε` from a Dataset
#####################################################################################
"""
    estimate_boxsizes(A::Dataset; kwargs...)
Return `k` exponentially spaced values: `base .^ range(lower + w, upper + z; length = k)`,
that are a good estimate for sizes ε that are used in calculating a [Fractal Dimension](@ref).

Let `d₋, d₊` be the minimum and maximum pair-wise distances in `A`, obtained
from [`minmax_pairwise_distance`](@ref).
Then `lower = log(base, d₋)` and `upper = log(base, d₊)`.
Because by default `w=1, z=-1`, we're providing sizes that are an order of mangitude
larger than the minimum distance, and an order of magnitude smaller than the minimum
distance.

## Keywords
* `w = 1, z = -1, k = 12` : as explained above.
* `metric = Euclidean()` : metric used in distance calculations.
* `base = 10.0` : the base used in the `log` function.
"""
function estimate_boxsizes(
        data::AbstractDataset;
        k::Int = 12, z = -1.0, w = 1.0, base = 10.0, metric = Euclidean()
    )

    min_d, max_d = minmax_pairwise_distance(A, metric)
    lower = log(b, min_dist)
    upper = log(b, max_d)

    if lower ≥ upper
        error(
        "Boxsize estimation failed: `upper` was found ≥ than `lower`. "*
        "Adjust keywords or provide a bigger dataset.")
    end
    if lower + w + 2 ≥ upper + z
        @warn "Boxsizes limits do not differ by 2 orders of magnitude or more. "*
        "Setting `w -= 0.5; z += 0.5`. Please adjust keywords or provide a bigger dataset."
        w -= 0.5; z += 0.5
    end

    return base .^ range(lower+w, upper+z; length = k)
end

"""
    minmax_pairwise_distance(A::Dataset, metric = Euclidean())
Return `min_d, max_d, min_pair, max_pair`: the minimum and maximum pairwise distance
of all points in the dataset, and the corresponding point pairs.
"""
function minmax_pairwise_distance(A::AbstractDataset, metric = Euclidean())
    tree = KDTree(A)
    min_d = eltype(A[1])(Inf)
    max_d = -min_d
    min_pair = max_pair = (0, 0)
    theiler = Theiler(0)
    for p in 1:length(A)
        inds, dists = Neighborhood.knn(tree, A[p], 1, theiler; sortds=false)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (p, ind)
        end
        if dist > max_d
            max_d = dist
            max_pair = (p, id)
        end
    end
    return min_d, max_d, min_pair, max_pair
end
