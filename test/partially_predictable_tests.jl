using ChaosTools
using Test, StaticArrays
using OrdinaryDiffEq: Vern9
using DynamicalSystemsBase.Systems: lorenz

ν_thresh_lower, ν_thresh_upper = 0.1, 0.9
C_thresh_lower, C_thresh_upper = 0.1, 0.9

println("\nTesting Partially predictable chaos...")

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
            @test chaos_type == :REG
            @test ν > ν_thresh_upper
            @test C > C_thresh_upper
        end
    end
end

@testset "Predictability Discrete" begin
    @testset "Henon map" begin
        ds = Systems.henon()
        a_reg = 0.8; a_ppc = 1.11; a_reg2 = 1.0; a_cha = 1.4
        res = [:REG, :REG, :PPC, :SC]

        for (i, a) in enumerate((a_reg, a_reg2, a_ppc, a_cha))
            set_parameter!(ds, 1, a)
            chaos_type, ν, C = predictability(ds; T_max = 400000)
            @test chaos_type == res[i]
        end
    end
end
