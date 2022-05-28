using ChaosTools
using ChaosTools.DynamicalSystemsBase
using OrdinaryDiffEq





# We have to define a callback to wrap the phase in [-π,π]
function affect!(integrator)
    uu = integrator.u
    D = length(integrator.u)
    for k in 1:D
        if integrator.u[k] < -π
            uu[k] = uu[k] + 2π
            set_state!(integrator, uu)
            u_modified!(integrator, true)
        elseif  integrator.u[k] > π
            uu[k] = uu[k] - 2π
            set_state!(integrator, uu)
            u_modified!(integrator, true)
        end
    end
end
function condition(u,t,integrator)
    D = length(integrator.u)
    for k in 1:D
        if (integrator.u[k] < -π  || integrator.u[k] > π)
            return true
        end
    end
    return false
end
cb = DiscreteCallback(condition,affect!)

D = 100
K = 3.; ω = range(-1, 1; length = D)
ds = Systems.kuramoto(D; K = K, ω = ω)
res = 200
default_diffeq = (reltol = 1e-9,  alg = Vern9(), callback = cb)
xg = range(-pi,pi,length = res)
grid = ntuple(x-> xg, D)

mapper = AttractorsViaRecurrences(ds, grid; diffeq = default_diffeq, Δt = 0.1, sparse = true)


nsamples = 100
bsn = zeros(Int16, nsamples)
@time for j in 1:nsamples
	u0 = rand(D) # Random Ic
	@show bsn[j] = mapper(u0)
end


# t1 = trajectory(ds, T, diffeq = default_diffeq)
# plot(range(0, T, step = 0.01), Matrix(t1))
#@save "kur_D9.jld2" D K ω bsn att grid
