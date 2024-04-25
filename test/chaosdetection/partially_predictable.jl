using ChaosTools
using Test
using Random

@inbounds function lorenz_rule!(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end

u0 = [0, 10.0, 0]
p0 = [10, 28, 8/3]
diffeq = (maxiters = 1e9,abstol = 1.0e-6, reltol = 1.0e-6)

lz = CoupledODEs(lorenz_rule!, u0, p0; diffeq)

ν_thresh_lower, ν_thresh_upper = 0.1, 0.9
C_thresh_lower, C_thresh_upper = 0.15, 0.9
println("\nTesting Partially predictable chaos...")

@testset "Predictability Lorenz" begin
    @testset "strongly chaotic" begin
        Random.seed!(12)
        set_parameter!(lz,2,180.70)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 1.22, T_max = 1e3, n_samples = 100,
        )
        @test chaos_type == :SC
        @test ν < ν_thresh_lower
        @test C < C_thresh_lower
        println("strongly chaotic: done")
    end
    @testset "PPC 1" begin
        Random.seed!(12)
        set_parameter!(lz,2,180.78)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.4, n_samples = 100, T_max = 400
        )
        @test chaos_type == :PPC
        @test ν < ν_thresh_lower
        @test C > C_thresh_upper
        println("ppc1: done")
    end
    @testset "PPC 2" begin
        Random.seed!(12)
        set_parameter!(lz,2,180.95)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.1, n_samples = 100, T_max = 400
        )
        @test chaos_type == :PPC
        @test ν < ν_thresh_lower
        @test C > C_thresh_upper
        println("ppc2: done")
    end
    @testset "laminar" begin
        Random.seed!(12)
        set_parameter!(lz,2,181.10)
        @time chaos_type, ν, C = predictability(lz;
            λ_max = 0.01, T_max = 400, n_samples = 100)
        @test chaos_type == :REG
        @test ν > ν_thresh_upper
        @test C > C_thresh_upper
        println("laminar: done")
    end
end

@testset "Predictability Discrete" begin
    @testset "Henon map" begin
        henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
		u0 = zeros(2)
		p0 = [1.4, 0.3]
		hn = DeterministicIteratedMap(henon_rule, u0, p0)
        
        a_reg = 0.8; a_ppc = 1.11; a_reg2 = 1.0; a_cha = 1.4
        res = [:REG, :REG, :PPC, :SC]

        for (i, a) in enumerate((a_reg, a_reg2, a_ppc, a_cha))
            set_parameter!(hn, 1, a)
            chaos_type, ν, C = predictability(hn; T_max = 100000)
            @test chaos_type == res[i]
        end
    end
end
