# Continuous implementation. I've tried numerous times to use `ContinuousCallback`,
# but it simply didn't work. Many different problems, and overall inaccuracy.
# Furthermore, it makes for much clearer, and faster, code, if we write our own.
# The callback pipeline is too obscure to know where things may fail.
# I've kept the callback source code at the end of this file for future reference.

ODEIntegrator = DynamicalSystemsBase.SciMLBase.DEIntegrator

#=
# Algorithm description for continuous systems
Alright, here's the plan. Trajectory is evolved iteratively via an integrator.
At each step, we first check how far away we are from u₀. If we are too far away,
we don't bother with any interpolation, because it is a very very costly operation.
Furthermore, we assumed that this algorithm will typically be used with rather
small sets. If we are close enough to the point, we then find
the point of trajectory closest to u0 with as much accuracy as possible.
After than, we find crossing times with linear interpolation or full-blown integrator
interpolation and root finding (we will offer both methods).

Once we have the crossing times, the code is actually the same as with discrete systems.

To find the closest point we do a minimization/optmization using the integrator
interpolation. On Slack, Chris suggested to use Nlopt. The derivative choice matters,
and while I don't think derivative methods here would be fast, Chris recommended AD.
In any case, I guess we can use the generic interface of Optimization.jl.
=#

function exit_entry_times(integ::ODEIntegrator, u0, εs, T;
        i, dmin, rootkw = (xrtol = 1e-12, atol = 1e-12)
    )

    maxε = εs[1]
    E = length(εs)
    reinit!(integ, u0)
    pre_outside = fill(false, E)      # `true` if outside the set. Previous step.
    cur_outside = copy(pre_outside)   # `true` if outside the set. Current step.
    prev_smallest_distance = -minimum(εs[end]) # this is a real number!
    curr_smallest_distance = prev_smallest_distance # same, but at current step
    exits   = [eltype(integ.t)[] for _ in 1:E]
    entries = [eltype(integ.t)[] for _ in 1:E]

    exit = integ.t
    τ, c = zero(exit), 0
    isoutside = false
    crossing(t) = signed_distance(integ(t), u0, ε)

    while integ.t < T
        step!(integ)
        # Check distance of uprev (because interpolation can happen only between
        # tprev and t) and if it is "too far away", then don't bother checking crossings.
        d = distance(integ.uprev, u0, ε)
        d > dmin && continue

        r = range(integ.tprev, integ.t; length = i)
        dp = signed_distance(integ.uprev, u0, ε) # `≡ crossing(r[1])`, crossing of previous step
        for j in 2:i
            dc = crossing(r[j])
            if dc*dp < 0 # the distances have different sign (== 0 case is dismissed)
                tcross = Roots.find_zero(crossing, (r[j-1], r[j]), ROOTS_ALG; rootkw...)
                if dp < 0 # here the trajectory goes from inside to outside
                    exit = tcross
                    break # if we get out of the box we don't check whether we go inside
                        # again: assume in 1 step we can't cross out two times
                else # otherwise the trajectory goes from outside to inside
                    exit > tcross && continue # scenario where exit was not calculated
                    τ += tcross - exit
                    c += 1
                    # here we don't `break` because we might cross out in the same step
                end
            end
            dp = dc
        end
        # Notice that the possibility of a trajectory going fully through the `ε`-set
        # within one of the interpolation points is NOT considered: simply increase
        # `interp_points` to increase accuracy and make this scenario possible.
    end
    return τ/c, c
end





using DynamicalSystemsBase.SciMLBase: ODEProblem, solve
# TODO: Notice that the callback methods are NOT used. They have problems
# that I have not been able to solve yet.
using DynamicalSystemsBase.SciMLBase: ContinuousCallback, CallbackSet

function exit_entry_times(ds::ContinuousDynamicalSystem, u0, εs, T;
        diffeq = NamedTuple(),
    )

    # error("Continuous system version is not yet ready.")

    eT = eltype(ds.t0)
    check_εs_sorting(εs, length(u0))
    exits = [eT[] for _ in 1:length(εs)]
    entries = [eT[] for _ in 1:length(εs)]

    # This callback fails, see https://github.com/SciML/DiffEqBase.jl/issues/580
    # callbacks = ContinuousCallback[]
    # for i in eachindex(εs)
    #     crossing(u, t, integ) = ChaosTools.signed_distance(u, u0, εs[i])
    #     negative_affect!(integ) = push!(entries[i], integ.t)
    #     positive_affect!(integ) = push!(exits[i], integ.t)
    #     cb = ContinuousCallback(crossing, positive_affect!, negative_affect!;
    #         save_positions = (false, false)
    #     )
    #     push!(callbacks, cb)
    # end
    # cb = CallbackSet(callbacks...)

    # This callback works, but it is only for 1 nesting level
    crossing(u, t, integ) = ChaosTools.signed_distance(u, u0, εs[1])
    negative_affect!(integ) = push!(entries[1], integ.t)
    positive_affect!(integ) = push!(exits[1], integ.t)
    cb = ContinuousCallback(crossing, positive_affect!, negative_affect!;
        save_positions = (false, false)
    )

    prob = ODEProblem(ds, (eT(0), eT(T)); u0 = u0, callback=cb)
    # we don't actually use the output of `solve`. It writes in-place to the arrays
    alg = DynamicalSystemsBase._get_solver(diffeq)

    solve(prob, alg;
        save_everystep = false, #dense = false,
        # DynamicalSystemsBase.CDS_KWARGS..., diffeq...
    )
    return exits, entries
end

function _default_dmin(εs)
    if εs[1] isa Real
        m = 4*maximum(εs)
    else
        m = 8*maximum(maximum(e) for e in εs)
    end
    return m
end

function mean_return_times(ds::ContinuousDynamicalSystem, u0, εs, T;
        i=10, dmin=_default_dmin(εs), diffeq = NamedTuple(), kwargs...
    )
    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    # This warning would be useful if the callback methods worked
    # if !haskey(diffeq, :alg) || diffeq[:alg] == DynamicalSystemsBase.DEFAULT_SOLVER
    #     error(
    #     "Please use a solver that supports callbacks using `OrdinaryDiffEq`. "*
    #     "For example `using OrdinaryDiffEq: Tsit5; mean_return_times(...; alg = Tsit5())`."
    #     )
    # end

    eT = eltype(ds.t0)
    check_εs_sorting(εs, length(u0))
    c = zeros(Int, length(εs)); τ = zeros(eT, length(εs))
    integ = integrator(ds, u0; diffeq)
    for j ∈ 1:length(εs)
        reinit!(integ)
        t = T isa AbstractVector ? T[j] : T
        μ = dmin isa AbstractVector ? dmin[j] : dmin
        ι = i isa AbstractVector ? i[j] : i
        τ[j], c[j] = mean_return_times_single(integ, u0, εs[j], t; i=ι, dmin=μ)
    end
    return τ, c
end

function mean_return_times_single(
        integ, u0, ε, T;
        i, dmin, rootkw = (xrtol = 1e-12, atol = 1e-12)
    )

    exit = integ.t
    τ, c = zero(exit), 0
    isoutside = false
    crossing(t) = signed_distance(integ(t), u0, ε)

    while integ.t < T
        step!(integ)
        # Check distance of uprev (because interpolation can happen only between
        # tprev and t) and if it is "too far away", then don't bother checking crossings.
        d = distance(integ.uprev, u0, ε)
        d > dmin && continue

        r = range(integ.tprev, integ.t; length = i)
        dp = signed_distance(integ.uprev, u0, ε) # `≡ crossing(r[1])`, crossing of previous step
        for j in 2:i
            dc = crossing(r[j])
            if dc*dp < 0 # the distances have different sign (== 0 case is dismissed)
                tcross = Roots.find_zero(crossing, (r[j-1], r[j]), ROOTS_ALG; rootkw...)
                if dp < 0 # here the trajectory goes from inside to outside
                    exit = tcross
                    break # if we get out of the box we don't check whether we go inside
                          # again: assume in 1 step we can't cross out two times
                else # otherwise the trajectory goes from outside to inside
                    exit > tcross && continue # scenario where exit was not calculated
                    τ += tcross - exit
                    c += 1
                    # here we don't `break` because we might cross out in the same step
                end
            end
            dp = dc
        end
        # Notice that the possibility of a trajectory going fully through the `ε`-set
        # within one of the interpolation points is NOT considered: simply increase
        # `interp_points` to increase accuracy and make this scenario possible.
    end
    return τ/c, c
end

# There seem to be fundamental problems with this method. It runs very slow,
# and does not seem to terminate........
# Furthermore, I have a suspision that calling a callback
# affects the trajectory solution EVEN IF NO MODIFICATION IS DONE to the `integ` object.
# I confirmed this by simply evolving the trajectory and looking at the plotting code
# at the `transit_time_tests.jl` file.
function mean_return_times_single_callbacks(
        ds::ContinuousDynamicalSystem, u0, ε, T;
        interp_points=10, diffeq...
    )

    eT = eltype(ds.t0)
    eeτc = zeros(eT, 4) # exit, entry, τ and count are the respective entries
    isoutside = Ref(false)

    crossing(u, t, integ) = ChaosTools.signed_distance(u, u0, ε)
    function positive_affect!(integ)
        # println("Positive affect (crossing out) at time $(integ.t) and state:")
        # println(integ.u)
        if isoutside[]
            return
        else
            # When crossing ε-set outwards, outside becomes true
            isoutside[] = true
            eeτc[1] = integ.t
            return
        end
    end
    function negative_affect!(integ)
        # println("Negative affect (crossing in) at time $(integ.t) and state:")
        # println(integ.u)
        # we have crossed the ε-set, but let's check whether we were already inside
        if isoutside[]
            isoutside[] = false
            eeτc[2] = integ.t
            eeτc[3] += eeτc[2] - eeτc[1]
            eeτc[4] += 1
            return
        else
            return
        end
    end
    cb = ContinuousCallback(crossing, positive_affect!, negative_affect!;
        save_positions = (false, false), interp_points,
    )

    prob = ODEProblem(ds, (eT(0), eT(T)); u0 = u0)
    sol = solve(prob;
        callback=cb, save_everystep = false, dense = false,
        save_start=false, save_end = false, diffeq...
    )
    return eeτc[3]/eeτc[4], round(Int, eeτc[4])
end
