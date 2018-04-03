using ChaosTools
using Base.Test


test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Estimate Delay" begin

    ds = Systems.henon()
    data = trajectory(ds,100;dt=1)
    x = data[:,1]
    @test estimate_delay(x,"first_zero") <= 2
    @test estimate_delay(x,"first_min")  <= 2
    @test estimate_delay(x,"exp_decay")  <= 2
    @test 3 <= estimate_delay(x,"mutual_inf"; k=1) <= 10
    @test 3 <= estimate_delay(x,"mutual_inf"; k=10) <= 10

    ds = Systems.roessler()
    dt = 0.01
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4
    @test 0.9 <= estimate_delay(x,"mutual_inf"; maxtau=200)*dt <= 1.3

    dt = 0.1
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4
    @test 1.3 <= estimate_delay(x,"mutual_inf")*dt <= 1.7
    @test 1.3 <= estimate_delay(x,"mutual_inf"; k=10)*dt <= 1.7

    ds = Systems.lorenz()
    dt = 0.05
    data = trajectory(ds,2000;dt=dt)
    x = data[500:end,1]
    @test 0.1 <= estimate_delay(x,"mutual_inf")*dt <= 0.3
    # println(estimate_delay(x,"exp_decay"))
    # #plot(autocor(x, 0:length(x)÷10, demean=true))
    # @test 2.5 <= estimate_delay(x,"exp_decay")*dt  <= 3.5
    #
    # dt = 0.1
    # data = trajectory(ds,2000;dt=dt)
    # x = data[:,1]
    # @test 2.5 <= estimate_delay(x,"exp_decay")*dt  <= 3.5
    # println(estimate_delay(x,"exp_decay"))

end


@testset "Estimate Dimension" begin
    s = sin.(0:0.1:1000)
    τ = 15
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.01) == 2


    ds = Systems.roessler();τ=15; dt=0.1
    data = trajectory(ds,1000;dt=dt)
    s = data[:,1]
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.1) ∈ [3, 4]

    ds = Systems.lorenz();τ=5; dt=0.01
    data = trajectory(ds,500;dt=dt)
    s = data[:,1]
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.1) ∈ [3, 4]
end
