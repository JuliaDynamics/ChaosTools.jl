using ChaosTools
using CairoMakie

function F!(du, u ,p, n)
    x,y,u,v = u
    A = 3; B = 0.3; C = 5.; D = 0.3; k = 0.4; 
    du[1] = A - x^2 + B*y + k*(x-u)
    du[2] = x
    du[3] = C - u^2 + D*v + k*(u-x)
    du[4] = u
    return 
end



# The region should not contain any attractors. 
R_min = [-4; -4.; -4.; -4.] 
R_max = [4.; 4.; 4.; 4.]


# Initial condition.
sampler, isinside = statespace_sampler(HRectangle(R_min,R_max))
x0 = sampler()

@show x0
df = DeterministicIteratedMap(F!, x0) 
xi = stagger_trajectory!(x0, df, isinside; δ = 2., Tm = 30) 
# @show Tp = escape_time!(xi, df, isinside)


v = stagger_and_step!(xi, df, 10000, isinside; stagger_mode = :adaptive, δ = 0.1, Tm = 10, max_steps = Int(1e4)) 
v = hcat(v...)'
scatter(v[:,1], v[:,3]; markersize = 3)

