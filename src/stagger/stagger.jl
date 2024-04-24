using DynamicalSystemsBase
# using ChaosTools
using LinearAlgebra:norm
using Random
# using Plots

function F!(du, u ,p, n)
    x,y,u,v = u
    A = 3; B = 0.3; C = 5.; D = 0.3; k = 0.4; 
    du[1] = A - x^2 + B*y + k*(x-u)
    du[2] = x
    du[3] = C - u^2 + D*v + k*(u-x)
    du[4] = u
    return 
end

function get_stagger!(x0, ds, δ, T, isinside; max_steps = Int(1e6))
    Tp = 0; xp = zeros(length(x0)); k = 1; mode = :exp
    if !isinside(x0)
        error("x0 must be in grid")
    end
    while Tp ≤ T 
        xp = x0 .+ rand_u(δ,length(x0); mode)
        while isinside(xp) == false
            xp = x0 .+ rand_u(δ,length(x0); mode)
        end

        if k > max_steps
           @show Tp, xp, x0, δ
           @warn "exp mode fails, δ is too small or Tm is too large"
           if mode == :unif 
              error("Uniform generator fails. Algorithm is stuck")
           end
           # if the exp mode is failing for some reason 
           # fallback on the uniform random generator 
           mode = :unif
           k = 1
        end
        Tp = escape_time!(xp, ds, isinside)
        k = k + 1
    end
    return xp, Tp
end

function stagger_trajectory(x0 ,ds, δ, Tm, isinside)
    T = escape_time!(x0, ds, isinside)
    xi = deepcopy(x0) 
    while T < Tm 
        xi, T = get_stagger!(xi, ds, δ, T, isinside)
    end
    return xi
end

function rand_u(δ, n; mode = :exp)
    if mode == :exp 
        a = -log10(δ)
        s = (15-a)*rand() + a
        u = (rand(n).- 0.5)
        u = u/norm(u)
        return u*10^-s
    else 
        return δ*(rand(n) .- 0.5)
    end
end

function stagger_and_step(x0 ,ds, δ, Tm, N, isinside)
    xi = stagger_trajectory(x0, ds, 1., Tm, isinside) 
    v = Vector{Vector{Float64}}(undef,N)
    v[1] = xi
    @show xi
    for n in 1:N
        if escape_time!(xi, ds, isinside) > Tm
            set_state!(ds, deepcopy(xi))
        else
            xp, Tp = get_stagger!(xi, ds, δ, Tm, isinside)
            set_state!(ds,deepcopy(xp))
        end 
        step!(ds)
        @show xi = get_state(ds)
        v[n] = xi
    end
    return v
end

function escape_time!(x0, ds, isinside) 
    x = deepcopy(x0) 
    set_state!(ds,x)
    ds.t = 1
    k = 1; max_step = 10000;
    while isinside(x) 
        k > max_step && break
        step!(ds)
        x = get_state(ds)
        k += 1
    end
    return ds.t
end
    


# function _is_in_grid(u, R_min, R_max) 
#     iswithingrid = true
#     @inbounds for i in eachindex(R_min)
#         if !(R_min[i] ≤ u[i] ≤ R_max[i])
#             iswithingrid = false
#             break
#         end
#     end
#     return iswithingrid
# end


# The region should not contain any attractors. 
R_min = [-4; -4.; -4.; -4.] 
R_max = [4.; 4.; 4.; 4.]


# Initial condition.
sampler, isinside = statespace_sampler(HRectangle(R_min,R_max))
x0 = sampler()

@show x0
df = DeterministicIteratedMap(F!, x0) 
xi = stagger_trajectory(x0, df, 2., 30, isinside) 
@show Tp = escape_time!(xi, df, isinside)


v = stagger_and_step(xi, df, 1e-10, 30, 50000, isinside) 
v = hcat(v...)'
scatter(v[:,1], v[:,3]; markersize = 3)
