export uncertainty_exponent

"""
    uncertainty_exponent(grid, basins; kwargs...) -> ε, f_ε ,α
Estimate the uncertainty exponent[^Grebogi1983] of the basins of attraction. This exponent
is related to the final state sensitivity of the trajectories in the phase space.
An exponent close to `1` means basins with smooth boundaries whereas an exponent close
to `0` represent complety fractalized basins, also called riddled basins.

`xg`, `yg` are 1-dim ranges that defines the grid of the initial conditions.
`basins` is the matrix containing the information of the basin.
This functionality is currently implemented only for 2D basins.

The output `f_ε` is a vector with the fraction of the balls of radius `ε` (in pixels)
that contain at least two initial conditions that lead to different attractors.
The ouput `α` is the estimation of the uncertainty exponent of the basins of attraction
by fitting a line in the `log.(f_ε)` vs `log.(ε)` curve, however it is recommended to
analyze the curve directly for more accuracy.

## Keyword arguments
* `precision = 1e-5` is the variance of the estimator of the uncertainty function.
  Values between `1e-7` and `1e-5` brings reasonable results.
* `max_ε = floor(Int, length(xg)/20)` is the maximum size in pixels of the ball to test.

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
related. We have `Δ₀ = 2 - α` where `Δ₀` is the box counting dimension,
see [Fractal Dimension](@ref).

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983
"""
function uncertainty_exponent(basins::Array; precision = 1e-5, max_ε = 10)

    n_dim = size(basins)
    r_ε = 3:max_ε # box size in pixels
    num_step = length(r_ε)
    N_u = zeros(Int, num_step) # number of uncertain box
    N = zeros(Int, num_step) # number of boxes
    ε = zeros(1, num_step) # resolution

    # Naive box counting
    for (k,eps) in enumerate(r_ε)
        completed = false;
        Nb, Nu = 0, 0
        # get indices of boxes
        box_indices = CartesianIndices(ntuple(i -> range(1,n_dim[i]-eps,step=eps), length(n_dim)))
        for box in box_indices
            ind = CartesianIndices(ntuple(i -> range(box[i],box[i]+eps-1,step=1), length(n_dim)))
            c = basins[ind]
            if length(unique(c))>1
                Nu = Nu + 1
            end
            Nb += 1
        end
        N_u[k] = Nu
        N[k] = Nb
        ε[k] = eps
    end
    f_ε = N_u ./ N
    # remove zeros in case there are any:
    ind = f_ε .> 0.0
    f_ε = f_ε[ind]
    ε = ε[ind]
    # get exponent via liner regression on `f_ε ~ ε^α`
    b, α = linreg(vec(log10.(ε)), vec(log10.(f_ε)))
    return ε, f_ε, α
end
