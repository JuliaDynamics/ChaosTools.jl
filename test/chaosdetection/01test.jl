using ChaosTools, Test

lo = Systems.logistic(r = 3.5)
reg = trajectory(lo, 10000; Ttr = 1000)
set_parameter!(lo, 1, 3.97)
cha = trajectory(lo, 10000)
testchaos01(cha)

println("\nTesting 0-1 chaos test...")
@testset "logistic" begin
    lo = Systems.logistic(r = 3.5)
    reg = trajectory(lo, 10000; Ttr = 1000)
    set_parameter!(lo, 1, 3.97)
    cha = trajectory(lo, 10000)

    @test testchaos01(cha)
    @test !testchaos01(reg)

    cs = range(0.1, π-0.1, length = 20)

    @test testchaos01(cha, cs)
    @test !testchaos01(reg, cs)

end


@testset "Standard Map" begin
    ds = Systems.standardmap(k = 0.9)
    cs = range(0.1, π-0.1, length = 20)
    @testset "chaotic" begin
        cha = trajectory(ds, 10000)[:, 1]
        @test testchaos01(cha)
        @test testchaos01(cha, cs)
    end
    @testset "regular" begin
        reg = trajectory(ds, 20000, [π, 0.2rand()])[:, 1]
        @test !testchaos01(reg)
        @test !testchaos01(reg, cs)
    end
end

using OrdinaryDiffEq

@testset "henonheiles" begin
    sp = [0, .295456, .407308431, 0] #stable periodic orbit: 1D torus
    qp = [0, .483000, .278980390, 0] #quasiperiodic orbit: 2D torus
    ch = [0, -0.25, 0.42081, 0] # chaotic orbit
    tt = 3000
    dt = 0.8 # <- very important!
    ds = Systems.henonheiles()
    diffeq = Dict(:Ttr => 10, :abstol=>1e-9, :reltol=>1e-9, :solver => Vern9())
    cs = range(0.1, π/3-0.1, length = 50)

    @testset "stable" begin
        tr = trajectory(ds, tt, sp; dt = dt, diffeq...)[:, 1]
        @test !testchaos01(tr)
        @test !testchaos01(tr, cs)
    end
    @testset "quasiperidic" begin
        tr = trajectory(ds, tt, qp; dt = dt, diffeq...)[:, 1]
        @test !testchaos01(tr)
        @test !testchaos01(tr, cs)
    end
    @testset "chaotic" begin
        tr = trajectory(ds, tt, ch; dt = dt, diffeq...)[:, 1]
        @test testchaos01(tr, cs)
    end
end
