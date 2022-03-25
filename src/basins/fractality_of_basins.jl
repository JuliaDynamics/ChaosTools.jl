export uncertainty_exponent, basins_fractal_dimension, basins_fractal_test, basin_entropy

"""
    basin_entropy(basins, ε = 20) -> Sb, Sbb
This algorithm computes the basin entropy `Sb` of the basins of attraction.
First, the input `basins`
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
have a fractal boundary, for a more precise test see [`basins_fractal_test`](@ref).
An important feature of the basin entropy is that it allows
comparisons between different basins using the same box size `ε`.

[^Daza2016]:
    A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán,
    Basin entropy: a new tool to analyze uncertainty in dynamical systems,
    Sci. Rep., 6, 31416, 2016.
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

In practice the algorithm computes the boundary basin entropy `Sbb` [`basin_entropy`](@ref)
for `Ntotal`
random balls of radius `ε`. If the computed value is equal to theoretical value of a smooth
boundary
(taking into account statistical errors and biases) then we decide that we have a smooth
boundary. Notice that the response `test_res` may depend on the chosen ball radius `ε`.
For larger size,
we may observe structures for smooth boundary and we obtain a *different* answer.

The output `test_res` is a symbol describing the nature of the basin and the output `Sbb` is
the estimated value of the boundary basin entropy with the sampling method.

[^Puy2021] Andreu Puy, Alvar Daza, Alexandre Wagemakers, Miguel A. F. Sanjuán. A test for
fractal boundaries based on the basin entropy. Commun Nonlinear Sci Numer Simulat, 95, 105588, 2021.

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
            N_stat[Nb] = bx_ent
        end
    end

    Ŝbb = mean(N_stat)
    σ_sbb = std(N_stat)/sqrt(Nb)
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


"""
    basins_fractal_dimension(basins; kwargs...) -> V_ε, N_ε ,d
Estimate the [Fractal Dimension](@ref) `d` of the boundary between basins of attraction using
the box-counting algorithm.

The output `N_ε` is a vector with the number of the balls of radius `ε` (in pixels)
that contain at least two initial conditions that lead to different attractors. `V_ε`
is a vector with the corresponding size of the balls. The ouput `d` is the estimation
of the box-counting dimension of the boundary by fitting a line in the `log.(N_ε)`
vs `log.(1/V_ε)` curve. However it is recommended to analyze the curve directly
for more accuracy.

## Keyword arguments
* `range_ε = 2:maximum(size(basins))÷20` is the range of sizes of the ball to
  test (in pixels).

## Description

It is the implementation of the popular algorithm of the estimation of the box-counting
dimension. The algorithm search for a covering the boundary with `N_ε` boxes of size
`ε` in pixels.
"""
function basins_fractal_dimension(basins::AbstractArray; range_ε = 3:maximum(size(basins))÷20)

    dims = size(basins)
    num_step = length(range_ε)
    N_u = zeros(Int, num_step) # number of uncertain box
    N = zeros(Int, num_step) # number of boxes
    V_ε = zeros(1, num_step) # resolution

    # Naive box counting estimator
    for (k,eps) in enumerate(range_ε)
        Nb, Nu = 0, 0
        # get indices of boxes
        bx_tuple = ntuple(i -> range(1, dims[i] - rem(dims[i],eps), step = eps), length(dims))
        box_indices = CartesianIndices(bx_tuple)
        for box in box_indices
            # compute the range of indices for the current box
            ind = CartesianIndices(ntuple(i -> range(box[i], box[i]+eps-1, step = 1), length(dims)))
            c = basins[ind]
            if length(unique(c))>1
                Nu = Nu + 1
            end
            Nb += 1
        end
        N_u[k] = Nu
        N[k] = Nb
        V_ε[k] = eps
    end
    N_ε = N_u
    # remove zeros in case there are any:
    ind = N_ε .> 0.0
    N_ε = N_ε[ind]
    V_ε = V_ε[ind]
    # get exponent via liner regression on `f_ε ~ ε^α`
    b, d = linreg(vec(-log10.(V_ε)), vec(log10.(N_ε)))
    return V_ε, N_ε, d
end

"""
    uncertainty_exponent(basins; kwargs...) -> ε, N_ε ,α
Estimate the uncertainty exponent[^Grebogi1983] of the basins of attraction. This exponent
is related to the final state sensitivity of the trajectories in the phase space.
An exponent close to `1` means basins with smooth boundaries whereas an exponent close
to `0` represent complety fractalized basins, also called riddled basins.

The output `N_ε` is a vector with the number of the balls of radius `ε` (in pixels)
that contain at least two initial conditions that lead to different attractors.
The ouput `α` is the estimation of the uncertainty exponent using the box-counting
dimension of the boundary by fitting a line in the `log.(N_ε)` vs `log.(1/ε)` curve.
However it is recommended to analyze the curve directly for more accuracy.

## Keyword arguments
* `range_ε = 2:maximum(size(basins))÷20` is the range of sizes of the ball to
  test (in pixels).

## Description

A phase space with a fractal boundary may cause a uncertainty on the final state of the
dynamical system for a given initial condition. A measure of this final state sensitivity
is the uncertainty exponent. The algorithm probes the basin of attraction with balls
of size `ε` at random. If there are a least two initial conditions that lead to different
attractors, a ball is tagged "uncertain". `f_ε` is the fraction of "uncertain balls" to the
total number of tries in the basin. In analogy to the fractal dimension, there is a scaling
law between, `f_ε ~ ε^α`. The number that characterizes this scaling is called the
uncertainty exponent `α`.

Notice that the uncertainty exponent and the box counting dimension of the boundary are
related. We have `Δ₀ = D - α` where `Δ₀` is the box counting dimension computed with
[`basins_fractal_dimension`](@ref) and `D` is the dimension of the phase space.
The algorithm first estimates the box counting dimension of the boundary and
returns the uncertainty exponent.

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity:
An obstruction to predictability, Physics Letters A, 99, 9, 1983
"""
function uncertainty_exponent(basins::AbstractArray; range_ε = 2:maximum(size(basins))÷20)
    V_ε, N_ε, d = basins_fractal_dimension(basins; range_ε)
    return V_ε, N_ε, length(size(basins)) - d
end
