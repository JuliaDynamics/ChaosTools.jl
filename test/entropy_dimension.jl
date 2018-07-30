using ChaosTools
using Test
using StatsBase

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting generalized entropy (genentropy) & linear scaling...")
@testset "Generalized Dimensions" begin
  @testset "Henon Map" begin
    ds = Systems.henon()
    ts = trajectory(ds, 200000)
    mat = Matrix(ts)
    # Test call with dataset
    genentropy(1, 0.001, ts)
    es = 10 .^ range(-0, stop = -3, length = 7)
    dd = zero(es)
    for q in [0,2,1, 2.56]
      for (i, ee) in enumerate(es)
        dd[i] = genentropy(q, ee, mat)
      end
      linr, dim = linear_region(-log.(es), dd)
      test_value(dim, 1.1, 1.3)
    end
  end
  @testset "Lorenz System" begin
    ds = Systems.lorenz()
    ts = trajectory(ds, 5000)
    es = 10 .^ range(1, stop = -3, length = 11)
    dd = zero(es)
    for q in [0,1,2]
      for (i, ee) in enumerate(es)
        dd[i] = genentropy(q, ee, ts)
      end
      linr, dim = linear_region(-log.(es), dd)
      if q == 0
        test_value(dim, 1.7, 1.9)
      else
        test_value(dim, 1.85, 2.0)
      end
    end
  end
end

println("\nTesting dimension calls (all names)...")
@testset "Dimension calls" begin
  ds = Systems.henon()
  ts = trajectory(ds, 20000)
  # Test call with dataset
  test_value(generalized_dim(1.32, ts), 1.1, 1.3)
  test_value(capacity_dim(ts), 1.1, 1.3)
  test_value(information_dim(ts), 1.1, 1.3)
end


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
        @test_throws ArgumentError permentropy([], order)
    end
end
