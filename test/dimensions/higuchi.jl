using ChaosTools
using Test
using Random

# Test sine wave (D ≈ 1)
N = 2^18
x = sin.(range(0, 1000; length = N))
Δ = higuchi(x)
@test Δ ≈ 1 atol=1e-2

# Test integrated gaussian noise (D ≈ 1.5)
x = cumsum(randn(Random.MersenneTwister(1234), N))
Δ = higuchi(x)
@test Δ ≈ 1.5 atol=1e-2
