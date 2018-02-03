using ChaosTools, PkgBenchmark

@benchgroup "lyapunov_DiscreteDS" ["Lyapunov"] begin
    henon = Systems.henon()
    @bench "λ_henon" lyapunov($henon, 10000, d0 = 1e-9, threshold = 1e-6, Ttr = 100)
end

@benchgroup "lyapunov_BigDiscreteDS" ["Lyapunov"] begin
    M = 5;
    ds = Systems.coupledstandardmaps(5)
    @bench "λ_CSM" lyapunov($ds, 10000, d0 = 1e-9, threshold = 1e-6, Ttr = 100)
end
