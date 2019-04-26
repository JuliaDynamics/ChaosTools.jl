using ChaosTools
using Test, StaticArrays
using OrdinaryDiffEq: Vern9
using DynamicalSystemsBase.Systems: lorenz

ν_thresh_lower, ν_thresh_upper = 0.05, 0.95
C_thresh_lower, C_thresh_upper = 0.05, 0.95

println("\nTesting predictability...")

@testset "Predictability Continuous" begin
    @testset "Lorenz map" begin
        @testset "Lorenz map - strongly chaotic" begin
            lz = lorenz(ρ=180.70)
            chaos_type, ν, C = predictability(lz; alg=Vern9(), maxiters=1e9)
            @test chaos_type == :SC
            @test ν < ν_thresh_lower
            @test C < C_thresh_lower
        end
        @testset "Lorenz map - PPC 1" begin
            lz = lorenz(ρ=180.78)
            chaos_type, ν, C = predictability(lz; alg=Vern9(), maxiters=1e9)
            @test chaos_type == :PPC
            @test ν < ν_thresh_lower
            @test C > C_thresh_upper
        end
        @testset "Lorenz map - PPC 2" begin
            lz = lorenz(ρ=180.95)
            chaos_type, ν, C = predictability(lz; alg=Vern9(), maxiters=1e9)
            @test chaos_type == :PPC
            @test ν < ν_thresh_lower
            @test C > C_thresh_upper
        end
        @testset "Lorenz map - laminar" begin
            lz = lorenz(ρ=181.10)
            chaos_type, ν, C = predictability(lz; T_max = 400, alg=Vern9(), maxiters=1e9)
            @test chaos_type == :LAM
            @test ν > ν_thresh_upper
            @test C > C_thresh_upper
        end
    end
end
