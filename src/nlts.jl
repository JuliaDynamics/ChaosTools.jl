using Neighborhood, StaticArrays, LinearAlgebra
using Distances: Metric, Cityblock, Euclidean
export NeighborNumber, WithinRange

export broomhead_king
export numericallyapunov
export Cityblock, Euclidean

#####################################################################################
#                    Numerical Lyapunov (from reconstruction)                       #
#####################################################################################
# Everything in this section is based on Ulrich Parlitz [1]

"""
    numericallyapunov(R::Dataset, ks;  refstates, w, distance, ntype)
Return `E = [E(k) for k ∈ ks]`, where `E(k)` is the average logarithmic distance
between states of a neighborhood that are evolved in time for `k` steps
(`k` must be integer).
Typically `R` is the result of delay coordinates of a single timeseries.

## Keyword Arguments

* `refstates = 1:(length(R) - ks[end])` : Vector of indices
  that notes which
  states of the reconstruction should be used as "reference states", which means
  that the algorithm is applied for all state indices contained in `refstates`.
* `w::Int = 1` : The [Theiler window](@ref).
* `ntype = NeighborNumber(1)` : The neighborhood type. Either [`NeighborNumber`](@ref)
  or [`WithinRange`](@ref). See [Neighborhoods](@ref) for more info.
* `distance::Metric = Cityblock()` : The distance function used in the
  logarithmic distance of nearby states. The allowed distances are `Cityblock()`
  and `Euclidean()`. See below for more info. The metric for finding neighbors is
  always the Euclidean one.


## Description
If the dataset exhibits exponential divergence of nearby states, then it should hold
```math
E(k) \\approx \\lambda\\cdot k \\cdot \\Delta t + E(0)
```
for a *well defined region* in the `k` axis, where ``\\lambda`` is
the approximated maximum Lyapunov exponent. ``\\Delta t`` is the time between samples in the
original timeseries. You can use [`linear_region`](@ref) with arguments `(ks .* Δt, E)` to
identify the slope (= ``\\lambda``) immediatelly, assuming you
have choosen sufficiently good `ks` such that the linear scaling region is bigger
than the saturated region.

The algorithm used in this function is due to Parlitz[^Skokos2016], which itself
expands upon Kantz [^Kantz1994]. In sort, for
each reference state a neighborhood is evaluated. Then, for each point in this
neighborhood, the logarithmic distance between reference state and neighborhood
state(s) is calculated as the "time" index `k` increases. The average of the above over
all neighborhood states over all reference states is the returned result.

If the `Metric` is `Euclidean()` then use the Euclidean distance of the
full `D`-dimensional points (distance ``d_E`` in ref.[^Skokos2016]).
If however the `Metric` is `Cityblock()`, calculate
the absolute distance of *only the first elements* of the `m+k` and `n+k` points
of `R` (distance ``d_F`` in ref.[^Skokos2016], useful when `R` comes from delay embedding).

[^Skokos2016]: Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 1 (section 1.3.2), Lecture Notes in Physics **915**, Springer (2016)

[^Kantz1994]: Kantz, H., Phys. Lett. A **185**, pp 77–87 (1994)
"""
function numericallyapunov(
        R::AbstractDataset{D, T}, ks;
        refstates = 1:(length(R) - ks[end]),
        w = 1,
        distance = Cityblock(),
        ntype = NeighborNumber(1),
    ) where {D, T}

    if ntype isa FixedMassNeighborhood
        @warn "`FixedMassNeighborhood` is deprecated in favor of `NeighborNumber`."
        ntype = NeighborNumber(ntype.k)
    end
    if ntype isa FixedSizeNeighborhood
        @warn "`FixedSizeNeighborhood` is deprecated in favor of `WithinRange`."
        ntype = WithinRange(ntype.k)
    end
    Ek = numericallyapunov(R, ks, refstates, Theiler(w), distance, ntype)
end

function numericallyapunov(
        R::AbstractDataset{D, T},
        ks::AbstractVector{Int},
        ℜ::AbstractVector{Int},
        theiler,
        distance::Metric,
        ntype::SearchType
    ) where {D, T}

    # ℜ = \Re<tab> = set of indices that have the points that one finds neighbors.
    # n belongs in ℜ and R[n] is the "reference state".
    # Thus, ℜ contains all the reference states the algorithm will iterate over.
    # ℜ is not estimated. it is given by the user. Most common choice:
    # ℜ = 1:(length(R) - ks[end])

    # ⩅(n) = \Cup<tab> = neighborhood of reference state n
    # which is evaluated for each n and for the given neighborhood type

    timethres = length(R) - ks[end]
    if maximum(ℜ) > timethres
        erstr = "Maximum index of reference states is > length(R) - ks[end] "
        erstr*= "and the algorithm cannot be performed on it. You have to choose "
        erstr*= "reference state indices of at most up to length(R) - ks[end]."
        throw(ArgumentError(erstr))
    end
    E = zeros(T, length(ks))
    E_n, E_m = copy(E), copy(E)
    tree = KDTree(R, Euclidean())
    skippedm = 0; skippedn = 0

    for n in ℜ
        # The ⋓(n) can be evaluated on the spot instead of being pre-calculated
        # for all reference states. Precalculating is faster, but allocates more memory.
        # Since ⋓[n] doesn't depend on `k` one can then interchange the loops:
        # Instead of k being the outermost loop, it becomes the innermost loop!
        point = R[n]
        ⋓ = isearch(tree, point, ntype, theiler(n))
        for m in ⋓
            # If `m` is nearer to the end of the timeseries than k allows
            # is it completely skipped (and length(⋓) reduced).
            if m > timethres
                skippedm += 1
                continue
            end
            for (j, k) in enumerate(ks) #ks should be small (of order 10 to 100 MAX)
                E_m[j] = log(delay_distance(distance, R, m, n, k))
            end
            E_n .+= E_m # no need to reset E_m
        end
        if skippedm >= length(⋓)
            skippedn += 1
            skippedm = 0
            continue # be sure to continue if no valid point!
        end
        E .+= E_n ./ (length(⋓) - skippedm)
        skippedm = 0
        fill!(E_n, zero(T)) #reset distances for n-th reference state
    end

    if skippedn >= length(ℜ)
        ers = "Skipped number of points ≥ length(R)...\n"
        ers*= "Could happen because all the neighbors fall within the Theiler "
        ers*= "window. Fix: increase neighborhood size."
        error(ers)
    end
    E ./= (length(ℜ) - skippedn)
end

@inline function delay_distance(di::Cityblock, R, m, n, k)
    @inbounds abs(R[m+k][1] - R[n+k][1])
end

@inline function delay_distance(di::Euclidean, R, m, n, k)
    @inbounds norm(R[m+k] - R[n+k])
end



#####################################################################################
#                                  Broomhead-King                                   #
#####################################################################################
"""
    broomhead_king(s::AbstractVector, d::Int) -> U, S, Vtr
Return the Broomhead-King coordinates of a timeseries `s`
by performing `svd` on high-dimensional embedding if `s` with dimension `d` with
minimum delay.

## Description
Broomhead and King coordinates is an approach proposed in [^Broomhead1987] that applies the
Karhunen–Loève theorem to delay coordinates embedding with smallest possible delay.

The function performs singular value decomposition
on the `d`-dimensional matrix ``X`` of ``s``,
```math
X = \\frac{1}{\\sqrt{N}}\\left(
\\begin{array}{cccc}
x_1 & x_2 & \\ldots & x_d \\\\
x_2 & x_3 & \\ldots & x_{d+1}\\\\
\\vdots & \\vdots & \\vdots & \\vdots \\\\
x_{N-d+1} & x_{N-d+2} &\\ldots & x_N
\\end{array}
\\right) = U\\cdot S \\cdot V^{tr}.
```
where ``x := s - \\bar{s}``.
The columns of ``U`` can then be used as a new coordinate system, and by
considering the values of the singular values ``S`` you can decide how many
columns of ``U`` are "important". See the documentation page for example application.

[^Broomhead1987]:  D. S. Broomhead, R. Jones and G. P. King, J. Phys. A **20**, 9, pp L563 (1987)
"""
function broomhead_king(x::AbstractArray, d::Int)
    X = trajectory_matrix(x, d)
    F = LinearAlgebra.svd(X)
    return F.U, F.S, F.Vt
end

function trajectory_matrix(x::AbstractArray, d::Int)
    xdash = mean(x)
    N = length(x); sqN = √N
    X = zeros(N-d+1, d)
    for j in 1:d
        for i in 0:(N-d)
            @inbounds X[i+1, j] = (x[j+i] - xdash)/sqN
        end
    end
    return X
end
