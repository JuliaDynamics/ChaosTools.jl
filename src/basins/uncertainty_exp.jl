export uncertainty_exponent

"""
    uncertainty_exponent(xg, yg, basins, integ; precision=1e-4, max_size=0) -> α,ε,f_ε
This function estimates the uncertainty exponent of the basins.

The ouput `α` is the estimation of the uncertainty exponent of the basins of attraction. This exponent is related to the final state sensitivity of the trajectories in the phase space. An exponent close to `1` means basins with smooth boundaries whereas an exponent close to `0` represent complety fractalized basins called a riddled basins.
The output `f_ε` is the ratio of the ball of radius `ε` that contains at least two initial conditions that lead to different attractors. `α` is the slope of the curve `f_ε` against `ε`. If the precision or the resolution is not high enough, the function gives a warning and a visual inspection of this curve is required in order to estimate the slope correctly.

Notice that the uncertainty exponent and the box counting dimension of the boundary are related. We have `d = 2 - α` where `d` is the box couting or capacity dimension.

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983

## Arguments
* `basin` : the matrix containing the information of the basin.
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions.

## Keyword arguments
* `precision` is the variance of the estimator of the uncertainty function. Values between 1e-7 and 1e-5 brings reasonable results.
* `max_size` is the maximum size in pixels of the ball to test.
"""
function uncertainty_exponent(xg,yg,basins; precision=1e-4, max_size=0)

    nx=length(xg)
    ny=length(yg)
    y_grid_res=yg[2]-yg[1]

    # resolution in pixels
    min_ε = 1;
    if max_size > 0
        max_ε = max_size
    else
        max_ε = floor(Int64,nx/20);
    end

    r_ε = min_ε:max_ε
    num_step=length(r_ε)
    N_u = zeros(Int64,1,num_step) # number of uncertain box
    N = zeros(Int64,1,num_step) # number of boxes
    ε = zeros(1,num_step) # resolution

    for (k,eps) in enumerate(r_ε)
        Nb=0; Nu=0; μ=0; σ²=0; M₂=0;
        completed = 0;
        # Find uncertain boxes
        while completed == 0
            kx = rand(1:nx)
            ky = rand(ceil(Int64,eps+1):floor(Int64,ny-eps))

            indy = range(ky-eps,ky+eps,step=1)
            c = [basins[kx,ky] for ky in indy]

            if length(unique(c))>1
                Nu = Nu + 1
            end
            Nb += 1

            # Welford's online average estimation and variance of the estimator
            M₂ = wel_var(M₂, μ, Nu/Nb, Nb)
            μ = wel_mean(μ, Nu/Nb, Nb)
            σ² = M₂/Nb

            # Stopping criterion: variance of the estimator of the mean bellow  precision
            if Nu > 50 && σ² < precision
                completed = 1
                @show Nu,Nb,σ²
            end

        end
        N_u[k]=Nu
        N[k]=Nb
        ε[k]=eps*y_grid_res
    end

    # uncertain function
    f_ε = N_u./N

    # remove zeros in case there are:
    ind = f_ε .> 0.
    f_ε =  f_ε[ind]
    ε = ε[ind]
    # get exponent
    # a,D =  linreg(vec(log10.(ε)), vec(log10.(f_ε)))
    D = linear_region(vec(log10.(ε)), vec(log10.(f_ε)))
    return D[2], vec(log10.(ε)), vec(log10.(f_ε))
end


function wel_var(M₂, μ, xₙ, n)
    μ₂ = μ + (xₙ - μ)/n
    M₂ = M₂ + (xₙ - μ)*(xₙ - μ₂)
    return M₂
end

function wel_mean(μ, xₙ, n)
    return μ + (xₙ - μ)/n
end
