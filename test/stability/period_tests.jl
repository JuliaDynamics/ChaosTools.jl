using ChaosTools, Test

Δt = 0.005

function test_period_algs(vs, ts, trueperiod, atol; methods = [
            :autocorrelation, :periodogram,
            :multitaper, :lombscargle, :zerocrossing
    ])

    for alg in methods
        @testset "$alg" begin
            Sys.WORD_SIZE == 32 && alg == :multitaper && continue
            @test estimate_period(vs, alg, ts) ≈ trueperiod atol = atol
        end
    end
end

@testset "sin(t)" begin
    tsin = 0:Δt:22π
    vsin = sin.(tsin)

    test_period_algs(vsin, tsin, 2π, 11.53Δt)
end

@testset "sin(2t) + 0.2cos(4t)" begin

    tsin = 0:Δt:22π
    vsin = sin.(2 .* tsin) + 0.2cos.(4 .* tsin)

    test_period_algs(vsin, tsin, π, 3Δt)
end
