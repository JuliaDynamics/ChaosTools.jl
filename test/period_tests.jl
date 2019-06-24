using ChaosTools, Test

println("\nTesting period estimation...")
dt = 0.01

@testset "simple sine" begin
tsin = 0:dt:22π
vsin = sin.(tsin)
vsin2 = sin.(2tsin)
@testset "ac" begin

L = length(tsin)÷10
p1 = estimate_period(vsin, "ac", tsin; L = L)
@test p1 ≈ 2π   atol = dt
p2 = estimate_period(vsin2, "ac", tsin; L = L)
@test p2 ≈  π   atol = dt

end
end

@testset "Roessler" begin
ds = Systems.roessler(ones(3))
T = 5000.0
tr = trajectory(ds, T; Ttr = 100.0, dt = dt)
v = tr[:, 1]
t = 0:dt:T

@testset "ac" begin
p = estimate_period(v, "ac", 0:dt:T)
# one oscillation of Roessler takes on average 6 time units
# from looking at the plot of v vs t
@test p ≈ 6  atol = 1
end

end


@testset "Modified FitzHugh-Nagumo"

function FHN(u, p, t)
    e, b, g = p
    v, w = u
    dv = min(max(-2 - v, v), 2 - v)
    dw = e*(v - g*w + b)
    return SVector(dv, dw)
end
fhn = ContinuousDynamicalSystem(FHN,SVector(-2, -0.6667),[0.04, 0, 0.8])
@testset "ac" begin
end

end
