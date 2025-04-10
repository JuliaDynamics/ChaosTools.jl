using ChaosTools, Test
using LinearAlgebra

henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon() = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])

#test if ensemble averaging gives the same as 
#the usual lyapunov exponent for autonomous system
@testset "time averaged and ensemble averaged lyapunov exponent" begin
    ds = henon()

    #eapd slope
    init_states = StateSpaceSet(0.2 .* rand(5000,2))
    ρ,times = ensemble_averaged_pairwise_distance(ds,init_states,100;Ttr=1000,sliding_param_rate_index=0)
    lyap_instant = lyapunov_instant(ρ,times;interval=10:20)

    #lyapunov exponent
    λ = lyapunov(ds,1000;Ttr=1000)
    @test isapprox(lyap_instant,λ;atol=0.01)
end

#test sliding Duffing map 
#-------------------------duffing stuff-----------------------
#https://doi.org/10.1016/j.physrep.2024.09.003

function duffing_drift(u0 = [0.1, 0.25]; ω = 1.0, β = 0.2, ε0 = 0.4, α=0.00045)
    return CoupledODEs(duffing_drift_rule, u0, [ω, β, ε0, α])
end

@inbounds function duffing_drift_rule(x, p, t)
    ω, β, ε0, α = p
    dx1 = x[2]
    dx2 = (ε0+α*t)*cos(ω*t) + x[1] - x[1]^3 - 2β * x[2]
    return SVector(dx1, dx2)
end

@testset "Duffing map" begin
    #----------------------------------hamiltonian case--------------------------------------
    duffing = duffing_drift(;β = 0.0,α=0.0,ε0=0.08) #no dissipation -> Hamiltonian case
    duffing_map = StroboscopicMap(duffing,2π)
    init_states_auto,_ = trajectory(duffing_map,5000,[-0.85,0.0];Ttr=0) #initial condition for a snapshot torus
    #set system to sliding
    set_parameter!(duffing_map,4,0.0005)

    ρ,times = ensemble_averaged_pairwise_distance(duffing_map,init_states_auto,100;Ttr=0,sliding_param_rate_index=4)
    lyap_instant = lyapunov_instant(ρ,times;interval=50:60)
    @test isapprox(lyap_instant,0.87;atol=0.01) #0.87 approximate value from article

    #-----------------------------------dissipative case------------------------------------
    duffing = duffing_drift() #no dissipation -> Hamiltonian case
    duffing_map = StroboscopicMap(duffing,2π)
    init_states = randn(5000,2) 
    ρ,times = ensemble_averaged_pairwise_distance(duffing_map,StateSpaceSet(init_states),100;Ttr=20,sliding_param_rate_index=4)
    lyap_instant = lyapunov_instant(ρ,times;interval=2:20)
    @test isapprox(lyap_instant,0.61;atol=0.01) #0.61 approximate value from article

end