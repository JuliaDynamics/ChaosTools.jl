using ChaosTools, Test, DynamicalSystemsBase

# Test `maximalexpansion`
@assert maximalexpansion([1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]) == 3.0 * 2.23606797749979 * 2.0
@assert maximalexpansion([1 0; 0 1]) == 1

# Test EEGraph on discrete dynamical systems.

tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)
tent_meanlist, tent_stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=60)

for (i, mean) in enumerate(tent_meanlist)
    @assert 0.6< mean/i < 0.8
end

expand2_eom(x, p, n) = 2x
expand2_jacob(x, p, n) = 2
expand2 = DiscreteDynamicalSystem(expand2_eom, 0.2, nothing, expand2_jacob)
expand_meanlist, expand_stdlist = EEgraph(expand2, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=10)

for (i, mean) in enumerate(expand_meanlist)
    @assert -0.1< mean/i < 0.1
end

lor = Systems.lorenz()
lor_gen() = [rand()*40-20, rand()*60-30, rand()*50]
lor_isinside(x) = -20 < x[1] < 20 && -30 < x[2] < 30 && 0 < x[3] < 50
@time lor_meanlist, lor_stdlist = EEgraph(lor, lor_gen, lor_isinside; batchcount=10, samplecount=100, steps=20, dT=1.0)

@assert 0.85 < (lor_meanlist[20] - lor_meanlist[1])/19 < 1.0

#####################################################################
# The following should be put into the documentation once this is finished.
#####################################################################

# Expand 2: is not chaotic, thus the plot should be a flat line.
expand2_eom(x, p, n) = 2x
expand2_jacob(x, p, n) = 2
expand2 = DiscreteDynamicalSystem(expand2_eom, 0.2, nothing, expand2_jacob)
@time meanlist, stdlist = EEgraph(expand2, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=10)
plot(meanlist, yerr=stdlist, leg=false)


# Modified tent map. This replicates Example B in Hunt and Ott.
tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)


# This replicates Figure 2.
@time tent_meanlist, tent_stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=60)
plot(tent_meanlist, yerr=tent_stdlist, leg=false)


# This replicates Figure 3.
@time meanlist, stdlist = EEgraph(tent, () -> rand()*2.5 - 1, x -> -1 < x < 1.5; batchcount=100, samplecount=100000, steps=60)
plot(meanlist, yerr=stdlist, leg=false)


# This replicates Figure 7
henon_iip = Systems.henon_iip(zeros(2); a = 4.2, b = 0.3)
henon_gen() = rand(2).*6 .- 3
henon_isinside(x) = -3<x[1]<3 &&  -3<x[2]<3
@time meanlist, stdlist = EEgraph(henon_iip, henon_gen, henon_isinside; batchcount=100, samplecount=100000, steps=25)
plot(meanlist, yerr=stdlist, leg=false)


# Exponential system
exp_eom(x, p, t) = x*p[1]
exp_u0 = [0.1]
exp_p = [1.0]
exp_jacob(x, p, t) = [1.0]
exp_sys = ContinuousDynamicalSystem(exp_eom, exp_u0, exp_p, exp_jacob)

exp_gen() = [rand()-0.5]
exp_isinside(x) = -10.0 < x[1] < 10.0
@time exp_meanlist, exp_stdlist = EEgraph(exp_sys, exp_gen, exp_isinside; batchcount=100, samplecount=10000, steps=20, dT=1.0)
plot((1:20)*1.0, exp_meanlist, yerr=exp_stdlist, leg=false)


#=
The Lorenz attractor is roughly bounded within the box [-20, 20]×[-30, 30]×[0, 50]
As can be seen in the plot, the Lorentz attractor has an expansion entropy of
about 0.92
=#

lor = Systems.lorenz()
lor_gen() = [rand()*40-20, rand()*60-30, rand()*50]
lor_isinside(x) = -20 < x[1] < 20 && -30 < x[2] < 30 && 0 < x[3] < 50
@time meanlist, stdlist = EEgraph(lor, lor_gen, lor_isinside; batchcount=100, samplecount=1000, steps=20, dT=1.0)
plot(meanlist, yerr=stdlist, leg=false)
plot!(x->0.92x+1.5, xlims=(0, 20))
