using ChaosTools, DynamicalSystemsBase, Test



println("\nTesting fixed points...")

@testset "Fixed points" begin
    @testset "Henon map" begin
        ds = Systems.henon()
        x = -1.5..1.5
        y = -0.5..0.5
        box = x × y
        
        fp, eigs, stable = fixedpoints(ds, box)
        
        # henon fixed points analytically
        hmfp(a, b) = (x = -(1-b)/2a - sqrt((1-b)^2 + 4a)/2a; return SVector(x, b*x))
        henon_fp = hmfp(ds.p[1], ds.p[2])
        
        @test size(fp) == (2,2)
        @test stable == [false, false]
        @test (henon_fp ≈ fp[1]) || (henon_fp ≈ fp[2])
    end

    @testset "standard map" begin
        ds = Systems.standardmap()
        x = 0.0..(2π - 100eps())
        box = x × x
        
        fp, eigs, stable = fixedpoints(ds, box)
        
        for (i, e) in enumerate(fp)
            if isapprox(e, SVector(0,0); atol = 1e-8)
                @test !stable[i]
            elseif e ≈ SVector(π, 0)
                @test stable[i]
            else
                @test false
            end
        end
    end

    @testset "Lorenz system" begin
        x = -20..20
        y = -20..20
        z = 0.0..40
        box = x × y × z
        σ = 10.0
        β = 8/3
        lorenzfp(ρ,β) = [
            SVector(0,0,0.0),
            SVector(sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1),
            SVector(-sqrt(β*(ρ-1)), -sqrt(β*(ρ-1)), ρ-1),
        ]

        stabil(σ, β) = σ*(σ+β+3)/(σ-β-1)

        @testset "ρ = $(ρ)" for (i, ρ) in enumerate((22, 28))
            ds = Systems.lorenz(; ρ, β, σ)
            afps = lorenzfp(ρ,β)
            fp, eigs, stable = fixedpoints(ds, box)
            fp = unique(fp.data)
            @test length(fp) == 3
            for p in fp
                @test any(w -> isapprox(w, p; atol = 1e-8), afps)
            end
            @test count(stable) == (i == 1 ? 2 : 0)
        end
    end
end
