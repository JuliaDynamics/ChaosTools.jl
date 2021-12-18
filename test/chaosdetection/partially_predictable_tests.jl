using ChaosTools
using Test
using Random

ν_thresh_lower, ν_thresh_upper = 0.1, 0.9
C_thresh_lower, C_thresh_upper = 0.15, 0.9
diffeqmaxit = (maxiters = 1e9,)
println("\nTesting Partially predictable chaos...")

@testset "Predictability Lorenz" begin
    @testset "strongly chaotic" begin
        Random.seed!(12)
        lz = Systems.lorenz(ρ=180.70)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 1.22, diffeq=diffeqmaxit, T_max = 1e3, n_samples = 100,
        )
        @test chaos_type == :SC
        @test ν < ν_thresh_lower
        @test C < C_thresh_lower
        println("strongly chaotic: done")
    end
    @testset "PPC 1" begin
        Random.seed!(12)
        lz = Systems.lorenz(ρ=180.78)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.4, diffeq=diffeqmaxit, n_samples = 100, T_max = 400
        )
        @test chaos_type == :PPC
        @test ν < ν_thresh_lower
        @test C > C_thresh_upper
        println("ppc1: done")
    end
    @testset "PPC 2" begin
        Random.seed!(12)
        lz = Systems.lorenz(ρ=180.95)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.1, diffeq=diffeqmaxit, n_samples = 100, T_max = 400
        )
        @test chaos_type == :PPC
        @test ν < ν_thresh_lower
        @test C > C_thresh_upper
        println("ppc2: done")
    end
    @testset "laminar" begin
        Random.seed!(12)
        lz = Systems.lorenz(ρ=181.10)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.01, T_max = 400, n_samples = 100, diffeq=diffeqmaxit
        )
        @test chaos_type == :REG
        @test ν > ν_thresh_upper
        @test C > C_thresh_upper
        println("laminar: done")
    end
end

@testset "Predictability Discrete" begin
    @testset "Henon map" begin
        ds = Systems.henon()
        a_reg = 0.8; a_ppc = 1.11; a_reg2 = 1.0; a_cha = 1.4
        res = [:REG, :REG, :PPC, :SC]

        for (i, a) in enumerate((a_reg, a_reg2, a_ppc, a_cha))
            set_parameter!(ds, 1, a)
            chaos_type, ν, C = predictability(ds; T_max = 100000)
            @test chaos_type == res[i]
        end
    end
end
