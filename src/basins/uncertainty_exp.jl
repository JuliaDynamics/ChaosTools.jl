export uncertainty_exponent

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
related. We have `Δ₀ = 2 - α` where `Δ₀` is the box counting dimension,
see [Fractal Dimension](@ref). The algorithm first estimates the box counting dimension of the
boundary and returns the uncertainty exponent.

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983
"""
function uncertainty_exponent(basins::AbstractArray; range_ε = 2:maximum(size(basins))÷20)

    n_dim = size(basins)
    num_step = length(range_ε)
    N_u = zeros(Int, num_step) # number of uncertain box
    N = zeros(Int, num_step) # number of boxes
    ε = zeros(1, num_step) # resolution

    # Naive box counting estimator
    for (k,eps) in enumerate(range_ε)
        Nb, Nu = 0, 0
        # get indices of boxes
        bx_tuple = ntuple(i -> range(1, n_dim[i] - rem(n_dim[i],eps), step = eps), length(n_dim))
        box_indices = CartesianIndices(bx_tuple)
        for box in box_indices
            # compute the range of indices for the current box
            ind = CartesianIndices(ntuple(i -> range(box[i], box[i]+eps-1, step = 1), length(n_dim)))
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
    f_ε = N_u
    # remove zeros in case there are any:
    ind = f_ε .> 0.0
    f_ε = f_ε[ind]
    ε = ε[ind]
    # get exponent via liner regression on `f_ε ~ ε^α`
    b, d = linreg(vec(-log10.(ε)), vec(log10.(f_ε)))
    α = length(n_dim) - d
    return ε, f_ε, α
end
