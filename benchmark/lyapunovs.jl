using ChaosTools, PkgBenchmark

@benchgroup "lyapunov_discrete" begin
        ds = Systems.henon()
        @bench "henon" lyapunov($ds, 10000,
        d0 = 1e-9, threshold = 1e-6, Ttr = 100)

        ds = Systems.towel()
        @bench "towel" lyapunov($ds, 10000,
        d0 = 1e-9, threshold = 1e-6, Ttr = 100)

        ds = Systems.coupledstandardmaps(5)
        @bench "CSM" lyapunov($ds, 10000,
        d0 = 1e-9, threshold = 1e-6, Ttr = 100)
    end


@benchgroup "lyapunov_continuous" begin
    ds = Systems.lorenz()
    @bench "lorenz" lyapunov($ds, 1000.0, d0 = 1e-9, threshold = 1e-6)
    @bench "lorenz2" lyapunov($ds, 1000.0, d0 = 1e-9, threshold = 1e-6)

    ds = Systems.duffing(β = -1, ω = 1, f = 0.3)
    @bench "duffing" lyapunov($ds, 1000.0, d0 = 1e-9, threshold = 1e-6)
    @bench "duffing2" lyapunov($ds, 1000.0, d0 = 1e-9, threshold = 1e-6)
end
