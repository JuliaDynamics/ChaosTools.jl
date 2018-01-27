if current_module() != ChaosTools
    using ChaosTools: permutation_entropy
end
using Base.Test

println("\nTesting permutation entropy...")
@testset "Permutation Entropy" begin
    @testset "Trivial entropies" begin
        @test permutation_entropy(zeros(10), 3) == 0
        @test permutation_entropy(ones(10), 3) == 0
        @test permutation_entropy(collect(1:10), 3) == 0
        @test permutation_entropy(collect(10:-1:1), 3) == 0
    end
    @testset "Examples" begin
        # Examples from Bandt & Pompe (2002):
        for (xs, order, desired) in [
                ([4, 7, 9, 10, 6, 11, 3], 2,
                 -(4/6)log2(4/6) - (2/6)log2(2/6)),
                ([4, 7, 9, 10, 6, 11, 3], 3,
                 -2(2/5)log2(2/5) - (1/5)log2(1/5)),
            ]
            @test permutation_entropy(xs, order; base=2) ≈ desired
        end
    end
    @testset "User Interface" begin
        order = Int(typemax(UInt8)) + 1
        try
            permutation_entropy([], order)
        catch err
            @test isa(err, ErrorException)
            @test contains(err.msg, "order = $order is too large")
        end
    end
end
