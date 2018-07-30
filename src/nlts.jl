using NearestNeighbors, StaticArrays, LinearAlgebra
using Distances: Metric, Cityblock, Euclidean

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
between states of a [`neighborhood`](@ref)
that are evolved in time for `k` steps (`k` must be integer).

## Keyword Arguments

* `refstates = 1:(length(R) - ks[end])` : Vector of indices
  that notes which
  states of the reconstruction should be used as "reference states", which means
  that the algorithm is applied for all state indices contained in `refstates`.
* `w::Int = 1` : The Theiler window, which determines
  whether points are separated enough in time to be considered separate trajectories
  (see [1] and [`neighborhood`](@ref)).
* `ntype::AbstractNeighborhood = FixedMassNeighborhood(1)` : The method to
  be used when evaluating the neighborhood of each reference state. See
  [`AbstractNeighborhood`](@ref) or [`neighborhood`](@ref) for more info.
* `distance::Metric = Cityblock()` : The distance function used in the
  logarithmic distance of nearby states. The allowed distances are `Cityblock()`
  and `Euclidean()`. See below for more info.


## Description
If the dataset/reconstruction
exhibits exponential divergence of nearby states, then it should clearly hold
```math
E(k) \\approx \\lambda\\Delta t k + E(0)
```
for a *well defined region* in the `k` axis, where ``\\lambda`` is
the approximated
maximum Lyapunov exponent. ``\\Delta t`` is the time between samples in the
original timeseries.
You can use [`linear_region`](@ref) with arguments `(ks .* Δt, E)` to
identify the slope
(= ``\\lambda``)
immediatelly, assuming you
have choosen sufficiently good `ks` such that the linear scaling region is bigger
than the saturated region.

The algorithm used in this function is due to Parlitz [1], which itself
expands upon Kantz [2]. In sort, for
each reference state a neighborhood is evaluated. Then, for each point in this
neighborhood, the logarithmic distance between reference state and neighborhood
state is
calculated as the "time" index `k` increases. The average of the above over
all neighborhood states over all reference states is the returned result.

If the `Metric` is `Euclidean()` then use the Euclidean distance of the
full `D`-dimensional points (distance ``d_E`` in ref. [1]).
If however the `Metric` is `Cityblock()`, calculate
the absolute distance of *only the first elements* of the `m+k` and `n+k` points
of the reconstruction `R` (distance ``d_F`` in ref. [1]).

## References

[1] : Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 1
(section 1.3.2), Lecture Notes in Physics **915**, Springer (2016)

[2] : Kantz, H., Phys. Lett. A **185**, pp 77–87 (1994)
"""
function numericallyapunov(R::AbstractDataset{D, T}, ks;
                           refstates = 1:(length(R) - ks[end]),
                           w = 1,
                           distance = Cityblock(),
                           ntype = FixedMassNeighborhood(1)) where {D, T}
    Ek = numericallyapunov(R, ks, refstates, w, distance, ntype)
end

function numericallyapunov(R::AbstractDataset{D, T},
                           ks::AbstractVector{Int},
                           ℜ::AbstractVector{Int},
                           w::Int,
                           distance::Metric,
                           ntype::AbstractNeighborhood) where {D, T}

    # ℜ = \Re<tab> = set of indices that have the points that one finds neighbors.
    # n belongs in ℜ and R[n] is the "reference state".
    # Thus, ℜ contains all the reference states the algorithm will iterate over.
    # ℜ is not estimated. it is given by the user. Most common choice:
    # ℜ = 1:(length(R) - ks[end])

    # ⩅(n) = \Cup<tab> = neighborhood of reference state n
    # which is evaluated for each n and for the given neighborhood ntype

    # Initialize:
    timethres = length(R) - ks[end]
    if maximum(ℜ) > timethres
        erstr = "Maximum index of reference states is > length(R) - ks[end] "
        erstr*= "and the algorithm cannot be performed on it. You have to choose "
        erstr*= "reference state indices of at most up to length(R) - ks[end]."
        throw(ArgumentError(erstr))
    end
    E = zeros(T, length(ks))
    E_n = copy(E); E_m = copy(E)
    data = R.data #tree data
    tree = KDTree(data, Euclidean()) # this creates a copy of `data`
    skippedm = 0; skippedn = 0

    for n in ℜ
        # The ⋓(n) can be evaluated on the spot instead of being pre-calculated
        # for all reference states. (it would take too much memory)
        # Since ⋓[n] doesn't depend on `k` one can then interchange the loops:
        # Instead of k being the outermost loop, it becomes the innermost loop!
        point = data[n]
        ⋓ = neighborhood(point, tree, ntype, n, w)
        for m in ⋓
            # If `m` is nearer to the end of the timeseries than k allows
            # is it completely skipped (and length(⋓) reduced).
            # If m is closer to n than the Theiler window allows, also skip.
            # The Theiler window defaults to τ
            if m > timethres || abs(m - n) <= w
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
        fill!(E_n, zero(T)) #reset distances for n reference state
    end

    if skippedn >= length(ℜ)
        ers = "skippedn ≥ length(R)\n"
        ers*= "Could happen because all the neighbors fall within the Theiler "
        ers*= "window. Fix: increase neighborhood size."
        error(ers)
    end
    E ./= length(ℜ) - skippedn
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
by performing `svd` on the so-called trajectory matrix with dimension `d`.

## Description
Broomhead and King coordinates is an approach proposed in [1] that applies the
Karhunen–Loève theorem to delay coordinates embedding with smallest possible delay.

The function performs singular value decomposition
on the `d`-dimensional trajectory matrix ``X`` of ``s``,
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

## References
[1] :  D. S. Broomhead, R. Jones and G. P. King, J. Phys. A **20**, 9, pp L563 (1987)
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
