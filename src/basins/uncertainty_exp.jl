export uncertainty_exponent

"""
    uncertainty_exponent(xg, yg, basins::Matrix; kwargs...) -> ε, f_ε ,α
Estimate the uncertainty exponent[^Grebogi1983] of the basins of attraction. This exponent
is related to the final state sensitivity of the trajectories in the phase space.
An exponent close to `1` means basins with smooth boundaries whereas an exponent close
to `0` represent complety fractalized basins, also called riddled basins.

`xg`, `yg` are 1-dim ranges that defines the grid of the initial conditions.
`basins` is the matrix containing the information of the basin, i.e. the output of
[`basins_map2D`](@ref) or [`basins_general`](@ref).

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
see [Fractal Dimension](@ref)

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity:
An obstruction to predictability, Physics Letters A, 99, 9, 1983
"""
function uncertainty_exponent(xg, yg, basins::Matrix;
        precision = 1e-5, max_ε = floor(Int, length(xg)/20),
    )

    nx, ny = length.((xg, yg))
    y_grid_res = yg[2] - yg[1]
    r_ε = 1:max_ε # resolution in pixels
    num_step = length(r_ε)
    N_u = zeros(Int, num_step) # number of uncertain box
    N = zeros(Int, num_step) # number of boxes
    ε = zeros(1, num_step) # resolution

    for (k,eps) in enumerate(r_ε)
        Nb, Nu, μ, σ², M₂ = 0, 0, 0, 0, 0
        completed = false;
        # Find uncertain boxes
        while !completed
            kx = rand(1:nx)
            ky = rand(ceil(Int,eps+1):floor(Int,ny-eps))
            indy = range(ky-eps,ky+eps,step=1)
            c = basins[kx, indy]
            if length(unique(c))>1
                Nu = Nu + 1
            end
            Nb += 1
            # Welford's online average estimation and variance of the estimator
            M₂ = wel_var(M₂, μ, Nu/Nb, Nb)
            μ = wel_mean(μ, Nu/Nb, Nb)
            σ² = M₂/Nb
            # Stopping criterion: variance of the estimator of the mean bellow precision
            if Nu > 50 && σ² < precision
                completed = true
            end
        end
        N_u[k] = Nu
        N[k] = Nb
        ε[k] = eps*y_grid_res
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


function wel_var(M₂, μ, xₙ, n)
    μ₂ = μ + (xₙ - μ)/n
    M₂ = M₂ + (xₙ - μ)*(xₙ - μ₂)
    return M₂
end

function wel_mean(μ, xₙ, n)
    return μ + (xₙ - μ)/n
end
