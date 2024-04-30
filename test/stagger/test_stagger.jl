using ChaosTools
using Test

@testset "Stagger Tests" begin

# Dynamical system with a saddle at 0
function F!(du, u ,p, n)
    x,y = u
    du[1] = x + y 
    du[2] = x - y
    return 
end


R_min = [-1.; -1.]; R_max = [1.; 1.]


# Initial condition.
sampler, isinside = statespace_sampler(HRectangle(R_min,R_max))
x0 = sampler()
df = DeterministicIteratedMap(F!, x0) 
xi = stagger_trajectory!(x0, df, isinside; δ = 2., Tm = 30) 
@test isinside(xi)
@test ChaosTools.escape_time!(xi, df, isinside) ≥ 30

# Test if all the points have escape time ≥ Tm 
# :exp mode
v = stagger_and_step!(xi, df, 10, isinside; δ = 1e-3, stagger_mode = :exp) 
for u in v
    @test ChaosTools.escape_time!(u, df, isinside) ≥ 30
end

# Test if all the points have escape time ≥ Tm 
# :unif mode
v = stagger_and_step!(xi, df, 10, isinside; δ = 1e-3, stagger_mode = :unif) 
for u in v
    @test ChaosTools.escape_time!(u, df, isinside) ≥ 30
end

# Test if all the points have escape time ≥ Tm 
# :adaptive mode
v = stagger_and_step!(xi, df, 10, isinside; δ = 1e-3, stagger_mode = :adaptive) 
for u in v
    @test ChaosTools.escape_time!(u, df, isinside) ≥ 30 - 1
end
end


