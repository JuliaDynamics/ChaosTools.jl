using ChaosTools, Test

println("\nTesting period estimation...")
# create signals
tsin = 0:0.1:22π
vsin = sin.(tsin)
vsin2 = sin.(2tsin)

@testset "autocorrelation" begin

L = length(tsin)÷10
p1 = estimate_period(vsin, "ac", tsin; L = L)
