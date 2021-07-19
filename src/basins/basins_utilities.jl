export match_attractors!, basin_entropy, basins_fractal_test


"""
    match_attractors!(b₋, a₋, b₊, a₊, [, method = :distance])
Attempt to match the attractors in basins/attractors `b₊, a₊` with those at `b₋, a₋`.
`b` is an array whose values encode the attractor ID, while `a` is a dictionary mapping
IDs to `Dataset`s containing the attractors (e.g. output of [`basins_general`](@ref)).
Typically the +,- mean after and before some change of parameter for a system.

In [`basins_general`](@ref) different attractors get assigned different IDs, however
which attractor gets which ID is somewhat arbitrary, and computing the basins of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs.

The modification of IDs is always done on the `b, a` that have less attractors.

`method` decides the matching process:
* `method = :overlap` matches attractors whose basins before and after have the most
  overlap (in pixels).
* `method = :distance` matches attractors whose state space distance the smallest.
"""
function match_attractors!(b₋, a₋, b₊, a₊, method = :distance)
    @assert size(b₋) == size(b₊)
    if length(a₊) > length(a₋)
        # Set it up so that modification is always done on `+` attractors
        a₋, a₊ = a₊, a₋
        b₋, b₊ = b₊, b₋
    end
    ids₊, ids₋ = sort!(collect(keys(a₊))), sort!(collect(keys(a₋)))
    if method == :overlap
        match_metric = _match_from_overlaps(b₋, a₋, ids₋, b₊, a₊, ids₊)
    elseif method == :distance
        match_metric = _match_from_distance(b₋, a₋, ids₋, b₊, a₊, ids₊)
    else
        error("Unknown method")
    end

    replacement_mapping = Dict{Int, Int}()
    for (i, ι) in enumerate(ids₊)
        v = match_metric[i, :]
        for j in sortperm(v) # go through the match metric in sorted order
            if ids₋[j] ∈ values(replacement_mapping)
                continue # do not use keys that have been used
            else
                replacement_mapping[ι] = ids₋[j]
            end
        end
    end

    # Do the actual replacing
    replace!(b₊, replacement_mapping...)
    aorig = copy(a₊)
    for (k, v) ∈ replacement_mapping
        a₊[v] = aorig[k]
    end
    # delete unused keys
    for k ∈ keys(a₊)
        if k ∉ values(replacement_mapping); delete!(a₊, k); end
    end
    return
end

function _match_from_overlaps(b₋, a₋, ids₋, b₊, a₊, ids₊)
    # Compute normalized overlaps of each basin with each other basin
    overlaps = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        Bi = findall(isequal(ι), b₊)
        for (j, ξ) in enumerate(ids₋)
            Bj = findall(isequal(ξ), b₋)
            overlaps[i, j] = length(Bi ∩ Bj)/length(Bj)
        end
    end
    overlaps
end

using LinearAlgebra
function _match_from_distance(b₋, a₋, ids₋, b₊, a₊, ids₊)
    closeness = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        aι = a₊[ι]
        for (j, ξ) in enumerate(ids₋)
            aξ = a₋[ξ]
            closeness[i, j] = 1 / minimum(norm(x .- y) for x ∈ aι for y ∈ aξ)
        end
    end
    closeness
end




"""
    basin_entropy(basins, ε = 20) -> Sb, Sbb
This algorithm computes the basin entropy `Sb` of the basins of attraction. First, the input `basins`
is divided regularly into n-dimensional boxes of side `ε` (along all dimensions).
Then `Sb` is simply the average of the Gibbs entropy computed over these boxes. The
function returns the basin entropy `Sb` as well as the boundary basin entropy `Sbb`.
The later is the average of the entropy only for boxes that contains at least two
different basins, that is, for the boxes on the boundary.

The basin entropy is a measure of the uncertainty on the initial conditions of the basins.
It is maximum at the value `log(n_att)` being `n_att` the number of attractors. In
this case the boundary is intermingled: for a given initial condition we can find
another initial condition that lead to another basin arbitriraly close. It provides also
a simple criterion for fractality: if the boundary basin entropy `Sbb` is above `log(2)`
then we have a fractal boundary. It doesn't mean that basins with values below cannot
have a fractal boundary, for a more precise test see [`basins_fractal_test`](@ref). An important feature of the basin entropy is that it allows
comparisons between different basins using the same box size `ε`.

[^Daza2016]: A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán, Basin entropy: a new tool to analyze uncertainty in dynamical systems, Sci. Rep., 6, 31416, 2016.

"""
function basin_entropy(basins, ε = 20)
    dims = size(basins)
    vals = unique(basins)
    pn = zeros(length(vals))
    Sb = 0; Nb = 0; N = 0
    bx_tuple = ntuple(i -> range(1, dims[i] - rem(dims[i],ε), step = ε), length(dims))
    box_indices = CartesianIndices(bx_tuple)
    for box in box_indices
        # compute the range of indices for the current box
        I = CartesianIndices(ntuple(i -> range(box[i], box[i]+ε-1, step = 1), length(dims)))
        box_values = [basins[k] for k in I]
        N = N + 1
        Nb = Nb + (length(unique(box_values)) > 1)
        Sb = Sb + _box_entropy(box_values)
    end
    return Sb/N, Sb/Nb
end

function _box_entropy(box_values)
    h = 0.
    for (k,v) in enumerate(unique(box_values))
        p = count( x -> (x == v), box_values)/length(box_values)
        h += p*log(1/p)
    end
    return h
end




"""
    basins_fractal_test(basins; ε = 20, Ntotal = 1000) -> test_res, Sbb
This is an automated test to decide if the boundary of the basins has fractal structures.
The bottom line is to look at the basins with a magnifier of size `ε` at random in `basins`.
If what we see in the magnifier looks like a smooth boundary (in average) we decide that
the boundary is smooth. If it is not smooth we can say that at the scale `ε` we have
structures, i.e., it is fractal.

In practice the algorithm computes the boundary basin entropy `Sbb` [`basin_entropy`](@ref) for `Ntotal`
random balls of radius `ε`. If the computed value is equal to theoretical value of a smooth boundary
(taking into account statistical errors and biases) then we decide that we have a smooth
boundary. Notice that the response `test_res` may depend on the chosen ball radius `ε`. For larger size,
we may observe structures for smooth boundary and we obtain a *different* answer.

The output `test_res` is a symbol describing the nature of the basin and the output `Sbb` is
the estimated value of the boundary basin entropy with the sampling method.

[^Puy2021] Andreu Puy, Alvar Daza, Alexandre Wagemakers, Miguel A. F. Sanjuán. A test for fractal boundaries based on the basin entropy. Commun Nonlinear Sci Numer Simulat, 95, 105588, 2021.

## Keyword arguments
* `ε = 20`: size of the ball for the test of basin. The result of the test may change with the size.
* `Ntotal = 1000`: number of balls to test in the boundary for the computation of `Sbb`
"""
function basins_fractal_test(basins; ε = 20, Ntotal = 1000)
    dims = size(basins)
    vals = unique(basins)
    S=Int(length(vals))
    pn=zeros(Float64,1,S)
    # Sanity check.
    if minimum(dims)/ε < 50
        @warn "Maybe the size of the grid is not fine enough."
    end
    if Ntotal < 100
        error("Ntotal must be larger than 1000 to gather enough statitics.")
    end

    v_pts = zeros(Float64, length(dims), prod(dims))
    I = CartesianIndices(basins)
    for (k,coord) in enumerate(I)
         v_pts[:, k] = [Tuple(coord)...]
    end
    tree = searchstructure(KDTree, v_pts, Euclidean())
    # Now get the values in the boxes.
    Nb = 1; N = 1; Sb = 0;
    N_stat = zeros(Ntotal)
    while Nb < Ntotal
        p = [rand()*(sz-ε)+ε for sz in dims]
        idxs = isearch(tree, p, WithinRange(ε))
        box_values = basins[idxs]
        bx_ent = _box_entropy(box_values)
        if bx_ent > 0
            Nb = Nb + 1
            Sb = Sb + bx_ent
            N_stat[Nb] = Sb/Nb
        end
        N = N + 1
    end

    Ŝbb = mean(N_stat[100:end])
    σ_sbb = std(N_stat[100:end])
    # Table of boundary basin entropy of a smooth boundary for dimension 1 to 5:
    Sbb_tab = [0.499999, 0.4395093, 0.39609176, 0.36319428, 0.33722572]
    if length(dims) ≤ 5
        Sbb_s = Sbb_tab[length(dims)]
    else
        Sbb_s = 0.898*length(dims)^-0.4995
    end
    # Systematic error aproximation for the disk of radius ε
    δub = 0.224*ε^-1.006

    tst_res = :smooth
    if Ŝbb < (Sbb_s - σ_sbb) ||  Ŝbb > (σ_sbb + Sbb_s + δub)
        println("Fractal boundary for size of box ε=", ε)
        tst_res = :fractal
    else
        println("Smooth boundary for size of box ε=", ε)
        tst_res = :smooth
    end

    return tst_res, Ŝbb
end
