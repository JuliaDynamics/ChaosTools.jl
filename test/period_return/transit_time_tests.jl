using ChaosTools, Test

println("\nTesting transit time statistics...")
@testset "Transit time statistics" begin

@testset "Standard map (exact)" begin
    # INPUT
    ds = Systems.standardmap()
    T = 10000 # maximum time

    # period 3 of standard map
    u0 = SVector(0.8121, 1.6243)
    # all first return times must be 1, and all second must be 2, and 3rd must be same as second
    εs = sort!([4.0, 2.0, 0.01]; rev=true)
    exits, entries = transit_time_statistics(ds, u0, εs, T)
    transits, returns = transit_return(exits, entries)

    @test all(x -> length(x) > 5, exits)
    @test all(x -> length(x) > 5, entries)
    @test all(issorted, exits)
    @test all(issorted, entries)
    @test all(isequal(2), transits[1])
    @test all(isequal(1), transits[2])
    @test transits[2] == transits[3]
    @test all(isequal(1), returns[1])
    @test all(isequal(2), returns[2])
    @test returns[2] == returns[3]

    # quasiperiodic around period 3:
    u0 = SVector(0.877, 1.565)
    εs = sort!([4.0, 0.5, 0.1]; rev=true)
    exits, entries = transit_time_statistics(ds, u0, εs, T)
    transits, returns = transit_return(exits, entries)

    @test all(issorted, exits)
    @test all(issorted, entries)
    @test all(x -> length(x) > 5, exits)
    @test all(x -> length(x) > 5, entries)

    # For ε=4.0, nothing changes with the before
    @test all(isequal(1), returns[1])
    @test all(isequal(2), transits[1])

    # Similarly, 0.5 should be the same as before
    @test all(isequal(1), transits[2])
    @test all(isequal(2), returns[2])

    # But now, the third entry is different, because it has the size of the quasiperiodic
    # stability island torous
    @test returns[3] ≠ returns[2]
    @test transits[3] ≠ transits[2]
    @test any(>(3), returns[3])
    @test all(isequal(1), transits[3]) # still need only one step to exit
end

@testset "Towel map (approximate)" begin

end
# ds = Systems.roessler()
# T = 1000.0 # maximum time
# u0 = trajectory(ds, 10; Ttr = 100)[end] # return center
# εs = sort!([0.1, 0.01, 0.001]; rev=true)
# diffeq = (;)

end
