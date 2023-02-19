using Distances: Euclidean
struct FirstElement end
using LinearAlgebra: norm
using Neighborhood

export NeighborNumber, WithinRange
export lyapunov_from_data
export Euclidean, FirstElement

"""
    lyapunov_from_data(R::Dataset, ks; kwargs...)

For the given dataset `R`, which is expected to represent a trajectory of a dynamical
system, calculate and return `E(k)`, which is the average logarithmic
distance between states of a neighborhood that are evolved in time for `k` steps
(`k` must be integer). The slope of `E` vs `k` approximates the maximum Lyapunov exponent.

Typically `R` is the result of delay coordinates embedding of a timeseries
(see DelayEmbeddings.jl).

## Keyword arguments

* `refstates = 1:(length(R) - ks[end])`: Vector of indices that notes which
  states of the dataset should be used as "reference states", which means
  that the algorithm is applied for all state indices contained in `refstates`.
* `w::Int = 1`: The [Theiler window](@ref).
* `ntype = NeighborNumber(1)`: The neighborhood type. Either [`NeighborNumber`](@ref)
  or [`WithinRange`](@ref). See [Neighborhoods](@ref) for more info.
* `distance = FirstElement()`: Specifies what kind of distance function is used in the
  logarithmic distance of nearby states. Allowed distances values are `FirstElement()`
  or `Euclidean()`, see below for more info. The metric for finding neighbors is
  always the Euclidean one.


## Description

If the dataset exhibits exponential divergence of nearby states, then it should hold
```math
E(k) \\approx \\lambda\\cdot k \\cdot \\Delta t + E(0)
```
for a *well defined region* in the ``k`` axis, where ``\\lambda`` is
the approximated maximum Lyapunov exponent. ``\\Delta t`` is the time between samples in the
original timeseries. You can use [`linear_region`](@ref) with arguments `(ks .* Δt, E)` to
identify the slope (= ``\\lambda``) immediatelly, assuming you
have choosen sufficiently good `ks` such that the linear scaling region is bigger
than the saturated region.

The algorithm used in this function is due to Parlitz[^Skokos2016], which itself
expands upon Kantz[^Kantz1994]. In sort, for
each reference state a neighborhood is evaluated. Then, for each point in this
neighborhood, the logarithmic distance between reference state and neighborhood
state(s) is calculated as the "time" index `k` increases. The average of the above over
all neighborhood states over all reference states is the returned result.

If the `distance` is `Euclidean()` then use the Euclidean distance of the
full `D`-dimensional points (distance ``d_E`` in ref.[^Skokos2016]).
If however the `distance` is `FirstElement()`, calculate
the absolute distance of *only the first elements* of the points of `R`
(distance ``d_F`` in ref.[^Skokos2016], useful when `R` comes from delay embedding).

[^Skokos2016]:
    Skokos, C. H. *et al.*, *Chaos Detection and Predictability* -
    Chapter 1 (section 1.3.2), Lecture Notes in Physics **915**, Springer (2016)

[^Kantz1994]: Kantz, H., Phys. Lett. A **185**, pp 77–87 (1994)
"""
function lyapunov_from_data(R::AbstractDataset{D, T}, ks;
        refstates = 1:(length(R) - ks[end]), w = 1,
        distance = FirstElement(), ntype = NeighborNumber(1),
    ) where {D, T}

    @assert all(k -> 0 ≤ k ≤ length(R), ks) "Invalid time range `ks`."
    timethres = length(R) - ks[end]
    if maximum(refstates) > timethres
        erstr = "Maximum index of reference states is > length(R) - ks[end]. "
        erstr*= "Choose indices of at most up to length(R) - ks[end]."
        throw(ArgumentError(erstr))
    end
    E = zeros(T, length(ks))
    E_n, E_m = copy(E), copy(E)
    tree = KDTree(R, Euclidean())
    skippedm = skippedn = 0
    theiler = Theiler(w)

    for n in refstates
        # The neighborhood(n) can be evaluated on the spot instead of being pre-calculated
        # for all reference states. This is actually faster than precalculating.
        # Since neighborhood(n) doesn't depend on `k` one can then interchange the loops:
        # Instead of k being the outermost loop, it becomes the innermost loop!
        neighborhood = isearch(tree, R[n], ntype, theiler(n))
        for m in neighborhood
            # If `m` is nearer to the end of the timeseries than k allows
            # is it completely skipped (no points to compute distance for)
            if m > timethres
                skippedm += 1
                continue
            end
            for (j, k) in enumerate(ks) #ks should be small (of order 10 to 100 MAX)
                E_m[j] = log(delay_distance(distance, R, m, n, k))
            end
            E_n .+= E_m
        end
        if skippedm ≥ length(neighborhood)
            skippedn += 1
            skippedm = 0
            continue # be sure to continue if no valid point!
        end
        E .+= E_n ./ (length(neighborhood) - skippedm)
        skippedm = 0
        E_n .= zero(T) # reset distances for n-th reference state
    end

    if skippedn ≥ length(refstates)
        ers = "Skipped number of points ≥ length(refstates)...\n"
        ers*= "Could happen because all the neighbors fall within the Theiler "
        ers*= "window. Fix: increase neighborhood size."
        error(ers)
    end
    E ./= (length(refstates) - skippedn)
end

@inline function delay_distance(::FirstElement, R, m, n, k)
    @inbounds abs(R[m+k][1] - R[n+k][1])
end

@inline function delay_distance(::Euclidean, R, m, n, k)
    @inbounds norm(R[m+k] - R[n+k])
end
