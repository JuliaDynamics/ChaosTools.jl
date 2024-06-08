using ChaosTools
using Test

# test if high period orbits are indeed periodic
@testset "Henon map" begin
    henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
    xs = range(0, stop = 2π, length = 5); ys = copy(xs)
    ics = [SVector(x,y) for x in xs for y in ys]
    o = 10
    m = 6
    fps = davidchacklai(ds, o, ics, m; abstol=1e-8, disttol=1e-12)
    tol = 1e-12
    for x0 in fps[end]
        set_state!(ds, x0)
        step!(ds, o)
        xn = current_state(ds)
        @test isapprox(x0, xn; atol = tol)
    end
end


# test analytical values and correct length
@testset "Logistic map" begin
    r = 1+sqrt(8)
    logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
    ds = DeterministicIteratedMap(logistic_rule, SVector(0.5), [r])
    xs = LinRange(0, 1, 5)
    ics = [SVector(x) for x in xs]
    o = 10
    m = 5
    fps = davidchacklai(ds, o, ics, m; abstol=1e-6, disttol=1e-13)
    tol = 1e-13

    ## known analytically
    @test length(fps[1]) == 2
    @test isapprox(SVector(0.0), minimum(fps[1]); atol = tol)
    @test isapprox(SVector((r-1)/r), maximum(fps[1]); atol = tol)

    sort!(fps[3])
    fps[3] = [round.(x, digits=7) for x in fps[3]]
    @test length(fps[3]) == 5
    @test isapprox(SVector(0.0), fps[3][1]; atol = tol)
    @test isapprox(SVector(0.1599288), fps[3][2]; atol = tol)
    @test isapprox(SVector(0.5143553), fps[3][3]; atol = tol)
    @test isapprox(SVector(0.7387961), fps[3][4]; atol = tol)
    @test isapprox(SVector(0.9563178), fps[3][5]; atol = tol)
    ##

    ## check if last periodic orbit indeed contains periodic points
    for x0 in fps[end]
        set_state!(ds, x0)
        step!(ds, o)
        xn = current_state(ds)
        @test isapprox(x0, xn; atol = tol)
    end
end