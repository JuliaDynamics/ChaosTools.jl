export uncertainty_exponent

"""
    uncertainty_exponent(xg, yg, basins, integ; precision=1e-3) -> α,ε,f_ε
This function estimates the uncertainty exponent of the boundary. It is related to the uncertainty dimension.

[C. Grebogi, S. W. McDonald, E. Ott, J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983]

## Arguments
* `basin` : the matrix containing the information of the basin.
* `xg`, `yg` : 1-dim range vector that defines the grid of the initial conditions.

## Keyword arguments
* `precision` variance of the estimator of the uncertainty function.

"""
function uncertainty_exponent(xg,yg,basins; precision=1e-4)

    nx=length(xg)
    ny=length(yg)

    # resolution in pixels
    min_ε = 1;
    max_ε = floor(Int64,nx/10);
    r_ε = min_ε:max_ε
    #r_ε = 10 .^ range(log10(min_ε),log10(max_ε),length=num_step)

    num_step=length(r_ε)
    N_u = zeros(1,num_step) # number of uncertain box
    N = zeros(1,num_step) # number of boxes
    ε = zeros(1,num_step) # resolution

    for (k,eps) in enumerate(r_ε)
        Nb=0; Nu=0; μ=0; σ²=0; M₂=0;
        completed = 0;
        # Find uncertain boxes
        while completed == 0
            kx = rand(1:nx)
            ky = rand(ceil(Int64,eps+1):floor(Int64,ny-eps))

            indy = range(ky-eps,ky+eps,step=1)
            c = [bsn.basin[kx,ky] for ky in indy]

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
                #@show Nu,Nb,σ²
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
    @. model(x, p) = p[1]*x+p[2]
    fit = curve_fit(model, vec(log10.(ε)), vec(log10.(f_ε)), [2., 2.])
    D = coef(fit)
    @show estimate_errors(fit)
    #D = linear_region(vec(log10.(ε)), vec(log10.(f_ε)))
    return D[1],vec(log10.(ε)), vec(log10.(f_ε))

end


function wel_var(M₂, μ, xₙ, n)

    μ₂ = μ + (xₙ - μ)/n
    M₂ = M₂ + (xₙ - μ)*(xₙ - μ₂)
    return M₂
end

function wel_mean(μ, xₙ, n)
    return μ + (xₙ - μ)/n
end
