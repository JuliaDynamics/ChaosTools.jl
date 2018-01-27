if current_module() != ChaosTools
    using ChaosTools: permentropy
end
using Base.Test

println("\nTesting permutation entropy...")
@testset "Permutation Entropy" begin
    @testset "Trivial entropies" begin
        @test permentropy(zeros(10), 3) == 0
        @test permentropy(ones(10), 3) == 0
        @test permentropy(collect(1:10), 3) == 0
        @test permentropy(collect(10:-1:1), 3) == 0
    end
    @testset "Examples" begin
        # Examples from Bandt & Pompe (2002):
        for (xs, order, desired) in [
                ([4, 7, 9, 10, 6, 11, 3], 2,
                 -(4/6)log2(4/6) - (2/6)log2(2/6)),
                ([4, 7, 9, 10, 6, 11, 3], 3,
                 -2(2/5)log2(2/5) - (1/5)log2(1/5)),
            ]
            @test permentropy(xs, order; base=2) ≈ desired
        end
    end
    @testset "User Interface" begin
        order = Int(typemax(UInt8)) + 1
        try
            permentropy([], order)
        catch err
            @test isa(err, ErrorException)
            @test contains(err.msg, "order = $order is too large")
        end
    end
end
