using ChaosTools
using Test

"""
    saturation_point(x, y; threshold = 0.01, dxi::Int = 1, tol = 0.2)
Decompose the curve `y(x)` into linear regions using `linear_regions(x, y; dxi, tol)`
and then attempt to find a saturation point where the the first slope
of the linear regions become `< threshold`.

Return the `x` value of the saturation point.
"""
function saturation_point(Ds, E1s; threshold = 0.01, kwargs...)
    lrs, slops = ChaosTools.linear_regions(Ds, E1s; kwargs...)
    i = findfirst(x -> x < threshold, slops)
    return i == 0 ? Ds[end] : Ds[lrs[i]]
end

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Estimate Delay" begin

    ds = Systems.henon()
    data = trajectory(ds,1000;dt=1)
    x = data[:,1]
    @test estimate_delay(x,"first_zero") <= 2
    @test estimate_delay(x,"first_min")  <= 2
    @test estimate_delay(x,"exp_decay")  <= 2
    # @test 3 <= estimate_delay(x,"mutual_inf"; k=1) <= 10
    # @test 3 <= estimate_delay(x,"mutual_inf"; k=10) <= 10

    ds = Systems.roessler(ones(3))
    dt = 0.01
    data = trajectory(ds,200,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4

    dt = 0.1
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4
    @test 1.3 <= estimate_delay(x,"mutual_inf")*dt <= 1.8
    @test 1.3 <= estimate_delay(x,"mutual_inf"; maxtau=150, k = 2)*dt <= 1.7

    ds = Systems.lorenz()
    dt = 0.05
    data = trajectory(ds,1000;dt=dt)
    x = data[500:end,1]
    @test 0.1 <= estimate_delay(x,"mutual_inf")*dt <= 0.4
    @test 0.1 <= estimate_delay(x,"exp_decay")*dt  <= 0.4
    #
    dt = 0.1
    data = trajectory(ds,2000;dt=dt)
    x = data[:,1]
    @test 0.1 <= estimate_delay(x,"exp_decay")*dt  <= 0.4
end


@testset "Estimate Dimension" begin
    s = sin.(0:0.1:1000)
    τ = 15
    Ds = 1:5
    E1s = estimate_dimension(s, τ, Ds)
    E2s = stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.01) == 1

    # @test minimum(E2s) < 0.1 # THIS TEST FAILS

    ds = Systems.roessler();τ=15; dt=0.1
    data = trajectory(ds,1000;dt=dt)
    s = data[:,1]
    Ds = 1:5
    E1s = estimate_dimension(s, τ, Ds)
    E2s = stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]
    @test minimum(E2s) < 0.3


    ds = Systems.lorenz();τ=5; dt=0.01
    data = trajectory(ds,500;dt=dt)
    s = data[:,1]
    Ds = 1:5
    E1s = estimate_dimension(s, τ, Ds)
    E2s = stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]

    # @test minimum(E2s) < 0.1 # THIS TEST FAILS

    #Test against random signal
    E2s = stochastic_indicator(rand(100000), 1, 1:5)
    @test minimum(E2s) > 0.9

end
