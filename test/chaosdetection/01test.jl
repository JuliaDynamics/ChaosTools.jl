using ChaosTools, Test

@inbounds function standardmap_rule(x, par, n)
    theta = x[1]; p = x[2]
    p += par[1]*sin(theta)
    theta += p
    while theta >= 2π; theta -= 2π; end
    while theta < 0; theta += 2π; end
    while p >= 2π; p -= 2π; end
    while p < 0; p += 2π; end
    return SVector(theta, p)
end
ds = DeterministicIteratedMap(standardmap_rule, [0.001, 0.008], [1.0])

cs = range(0.1, π-0.1, length = 20)
@testset "chaotic" begin
    cha = trajectory(ds, 10000)[1][:, 1]
    @test testchaos01(cha)
    @test testchaos01(cha, cs)
end
@testset "regular" begin
    reg = trajectory(ds, 20000, [π, 0.01])[1][:, 1]
    @test !testchaos01(reg)
    @test !testchaos01(reg, cs)
end
