using OrdinaryDiffEq
using DynamicalSystemsBase
using ChaosTools

ds = Systems.lorenz()

tim = time()

lyapunovs(ds, 2000)

tim = time() - tim

if typeof(ds) <: DiscreteDynamicalSystem
    println("Discrete system. ")
else
    println("Default solver: ", DynamicalSystemsBase.CDS_KWARGS.alg)
end
println("Time to first run of `lyapunov` ( ≡ compile time):")
println(tim, " seconds.\n")

# Results:
# Default solver: SimpleDiffEq.SimpleATsit5()
# Time to first run of `lyapunov` ( ≡ compile time):
# 7.906000137329102 seconds.
#
# Default solver: Tsit5()
# Time to first run of `lyapunov` ( ≡ compile time):
# 13.582000017166138 seconds.
#
# Default solver: Vern9(true)
# Time to first run of `lyapunov` ( ≡ compile time):
# 24.50499987602234 seconds.
