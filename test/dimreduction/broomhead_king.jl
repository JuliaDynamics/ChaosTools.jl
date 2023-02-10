using ChaosTools
using Test

X = [sin(t) for t in 1:0.1:10π]

U, S, Vtr = broomhead_king(X, 10)

@test all(>(1e-6), S[1:3])
@test all(<(1e-6), S[4:end])