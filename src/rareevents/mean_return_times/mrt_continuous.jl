#=
Continuous implementation. I've tried numerous times to use `ContinuousCallback`,
but it simply didn't work. Many different problems, and overall inaccuracy.
Furthermore, it makes for much clearer, and faster, code, if we write our own.
The callback pipeline is too obscure to know where things may fail.
I've kept the callback source code at the end of this file for future reference.

The docstring of `exit_entry_times` describes how the current setup works.

To find the closest point we do a minimization/optmization using the integrator
For minimization Chris suggested to use Nlopt (on Slack). The derivative choice matters,
and Chris recommended AD. We can use the generic interface of Optimization.jl.

However, it appears to me that the interface provided by Optim.jl for univariate functions:
https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval
is much simpler and seems to be exactly what we need... So that's what I'll use!
(ALso, it is very well documented, which is what we want for source code clarity)
=#
import ProgressMeter
##########################################################################################
# Main function
##########################################################################################
"Specification for continuous systems in [`exit_entry_times`](@ref)."
struct CrossingLinearIntersection end

"Specification for continuous systems in [`exit_entry_times`](@ref)."
Base.@kwdef struct CrossingAccurateInterpolation
    abstol::Float64 = 1e-12 # these are converted to the (different) keywords
    reltol::Float64 = 1e-6  # that Optim.jl and Roots.jl take
end

function exit_entry_times(integ::AbstractODEIntegrator, u0, εs, T;
        crossing_method = CrossingLinearIntersection(),
        threshold_multiplier = Inf, # This is an undocumented keyword
        threshold_distance = _default_threshold_distance(εs, threshold_multiplier),
        show_progress = true,
    )
    metric = eltype(εs) <: Real ? Euclidean() : Chebyshev()
    if metric isa Chebyshev
        error("""
        Hyper-rectangles are not yet supported in continous systems. If you need this,
        please make a PR that tests, and corrects, the distance logic with hyper-rectangles!
        """)
        # The code will error or do wonky stuff with rectangles, I haven't had the time
        # to test it thoroughly...
    end
    E = length(εs)
    prev_outside = fill(false, E)      # `true` if outside the set. Previous step.
    curr_outside = copy(prev_outside)  # `true` if outside the set. Current step.
    exits   = [eltype(integ.t)[] for _ in 1:E]
    entries = [eltype(integ.t)[] for _ in 1:E]
    prog = ProgressMeter.Progress(
        round(Int, T); desc="Exit-entry times:", enabled=show_progress
    )
    t0 = integ.t
    while (integ.t - t0) < T
        step!(integ)
        ProgressMeter.update!(prog, round(Int, integ.t - t0))
        # Check whether we are too far away from the point to bother doing anything
        # TODO: I wonder if we can use the previous minimum distance and compare it
        # with current one for accelerating the search...?
        curr_distance = evaluate(metric, get_state(integ), u0)
        curr_distance > threshold_distance && continue
        # Obtain mininum distance and check which is the outermost box we are out of
        tmin, dmin = closest_trajectory_point(integ, u0, metric, crossing_method)
        out_idx = first_outside_index(get_state(integ), u0, εs, E)
        out_idx_min = first_outside_index(dmin, εs)
        # if we were outside all, and min distance also outside all, we skip
        out_idx_min == 1 && all(prev_outside) && continue
        # something changed, compute state, interpolate, and update
        curr_outside[out_idx:end] .= true
        curr_outside[1:(out_idx - 1)] .= false
        # Depending on the method, different information is used to find crossings
        if crossing_method isa CrossingLinearIntersection
            update_exits_and_entries_linear!(
                exits, entries, integ, u0, εs, prev_outside, curr_outside
            )
        elseif crossing_method isa CrossingAccurateInterpolation
            update_exits_and_entries_interpolation!(
                exits, entries, out_idx, prev_outside, curr_outside,
                integ, u0, εs, tmin, crossing_method, out_idx_min
            )
        end
        prev_outside .= curr_outside
    end
    ProgressMeter.finish!(prog)
    return exits, entries
end

function first_return_time(integ::AbstractODEIntegrator, u0, ε, T;
        crossing_method = CrossingLinearIntersection(),
        show_progress = true,
    )
    metric = ε isa Real ? Euclidean() : Chebyshev()
    if metric isa Chebyshev
        error("""
        Hyper-rectangles are not yet supported in continous systems. If you need this,
        please make a PR that tests, and corrects, the distance logic with hyper-rectangles!
        """)
        # The code will error or do wonky stuff with rectangles, I haven't had the time
        # to test it thoroughly...
    end

    prog = ProgressMeter.Progress(
        round(Int, T); desc="First return time:", enabled=show_progress
    )
    t0 = integ.t
    # First step until exiting the set
    tmin, dmin = t0, zero(eltype(get_state(integ)))
    isout = false
    while !isout
        step!(integ)
        tmin, dmin = closest_trajectory_point(integ, u0, metric, crossing_method)
        isout = isoutside(dmin, ε)
    end
    # Now iterate until we are back in the set again
    while (integ.t - t0) < T
        step!(integ)
        ProgressMeter.update!(prog, round(Int, integ.t - t0))
        # Obtain mininum distance and check if we are out the box
        tmin, dmin = closest_trajectory_point(integ, u0, metric, crossing_method)
        isout = isoutside(dmin, ε)
        if !isout
            ProgressMeter.finish!(prog)
            return tmin - t0
        end
    end
    return NaN # in case it didn't return during time `T`
end

##########################################################################################
# CrossingLinearIntersection version
##########################################################################################
function closest_trajectory_point(integ, u0, metric, method::CrossingLinearIntersection)
    if !(metric isa Euclidean)
        error("Linear interpolation for crossing times only works with spheres (real `ε`).")
    end
    origin, endpoint = integ.uprev, integ.u
    # from https://diego.assencio.com/?index=ec3d5dfdfc0b6a0d147a656f0af332bd
    x = endpoint .- origin
    x² = x ⋅ x
    λ = ((u0 .- origin) ⋅ x) / x²
    if λ < 0
        umin, tmin = origin, integ.tprev
    elseif λ > 1
        umin, tmin = endpoint, integ.t
    else
        umin, tmin = origin .+ λ*x, integ.tprev + λ*(integ.t - integ.tprev)
    end
    dmin = euclidean(umin, u0)
    return tmin, dmin
end

function update_exits_and_entries_linear!(
        exits, entries, integ, u0, εs, pre_outside, cur_outside
    )
    # In this method we iterate for exits and entries at the same time, because we can
    # efficiently find both entry and exit if it happens to be within the current step
    # Notice that in this method we also don't really use the already found time of
    # minimum, because the actual numerical operations don't reduce even if we know it
    origin, endpoint = integ.uprev, integ.u
    tprev, tcurr = integ.tprev, integ.t
    for j in eachindex(exits)
        radius = εs[j]
        inter = line_hypersphere_intersection(u0, radius, origin, endpoint)
        isnothing(inter) && continue # no intersections for this radius
        t1, t2 = inter # t1 is the smallest number!
        cross1, cross2 = (0 ≤ t1 ≤ 1), (0 ≤ t2 ≤ 1)
        if cross1 && cross2 # we cross the entire circle within the step!
            push!(entries[j], tprev + t1*(tcurr - tprev))
            push!(exits[j], tprev + t2*(tcurr - tprev))
        # Either one or no crossings within current time range
        elseif cross2 # we're crossing out
            # @assert cur_outside[j] && !pre_outside[j] # this is de facto true
            push!(exits[j], tprev + t2*(tcurr - tprev))
        elseif cross1 # we're crossing in
            # @assert pre_outside[j] && !cur_outside[j] # this is de facto true
            push!(entries[j], tprev + t1*(tcurr - tprev))
        end
    end
end

using LinearAlgebra: dot, normalize
"""
    line_hypersphere_intersection(center, radius, origin, endpoint)
Return the intersections `t1, t2` of the line defined between `origin, endpoint` and
the given hypersphere. `t1, t2` are normalized with 0 meaning the origin, and
1 meaning the endpoint. If no intersections exist, return `nothing`.

Uses the formulas from: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection .
"""
function line_hypersphere_intersection(center, radius, origin, endpoint)
    direction = endpoint .- origin
    c, r, o, u = center, radius, origin, direction # for formulas like Wikipedia
    unorm2 = dot(u, u)
    oc = o .- c
    x = dot(u, oc)
    ∇ = x^2 - unorm2*(dot(oc, oc) - r^2)
    ∇ ≤ 0 && return nothing
    sq∇ = sqrt(∇)
    return (-x - sq∇, -x + sq∇) ./ unorm2
end


##########################################################################################
# CrossingAccurateInterpolation version
##########################################################################################
import Optim, Roots
function closest_trajectory_point(integ, u0, metric, method::CrossingAccurateInterpolation)
    # use Optim.jl to find minimum of the function
    f = (t) -> evaluate(metric, integ(t), u0)
    # Then find minimum of `f` in limits `(tprev, t)`
    optim = Optim.optimize(
        f, integ.tprev, integ.t, Optim.Brent();
        store_trace=false, abs_tol = method.abstol, rel_tol = method.reltol,
    )
    # You can show `Optim.iterations(optim)` to get an idea of convergence.
    # @show Optim.iterations(optim)
    tmin, dmin = Optim.minimizer(optim), Optim.minimum(optim)
    return tmin, dmin
end

function update_exits_and_entries_interpolation!(
        exits, entries, out_idx, pre_outside, cur_outside,
        integ, u0, εs, tmin0, method, out_idx_min
    )
    tprev, tcurr = integ.tprev, integ.t
    # The function needs three clauses: one for crossing through the entire set,
    # one for crossing the exits only, and one for crossing the entries only.

    # Crossing out; we are iterating from smallest set to larger, to update tmin!
    tmin = tmin0
    @inbounds for j in length(pre_outside):-1:out_idx
        # Check if we actually exit `j` set
        cur_outside[j] && !pre_outside[j] || continue
        # Perform rootfinding to find crossing point accurately
        crossing = (t) -> signed_distance(integ(t), u0, εs[j])
        tcross = Roots.find_zero(
            crossing, (tmin, tcurr), Roots.A42();
            atol = method.abstol, rtol = method.reltol
        )
        push!(exits[j], tcross)
        # update tmin, which now is the time to exit the previous inner set
        tmin = tcross
    end

    # Crossing in; we are iterating from largest set to smallest, to update tmin!
    tmin = tmin0
    @inbounds for j in 1:(out_idx - 1)
        # Check if we actually enter `j` set
        pre_outside[j] && !cur_outside[j] || continue
        # Perform rootfinding to find crossing point accurately
        crossing = (t) -> signed_distance(integ(t), u0, εs[j])
        tcross = Roots.find_zero(
            crossing, (tprev, tmin), Roots.A42();
            atol = method.abstol, rtol = method.reltol
        )
        push!(entries[j], tcross)
        # update `tmin`, which now is the time to enter the previous outer set
        tmin = tcross
    end

    # Last clause: checks for crossing through the entire set
    if out_idx_min > 1 # minimum possible distance is inside at least one set
        t1, t2 = tprev, tcurr # crossing times, will be updated later!
        @inbounds for j in 1:(out_idx_min - 1)
            pre_outside[j] && cur_outside[j] || continue # ensure that we are for sure out
            # First find crossing in
            crossing = (t) -> signed_distance(integ(t), u0, εs[j])
            tcross = Roots.find_zero(
                crossing, (t1, tmin0), Roots.A42();
                atol = method.abstol, rtol = method.reltol
            )
            push!(entries[j], tcross)
            t1 = tcross
            # Then find crossing out
            tcross = Roots.find_zero(
                crossing, (tmin0, t2), Roots.A42();
                atol = method.abstol, rtol = method.reltol
            )
            push!(exits[j], tcross)
            t2 = tcross
        end
    end
end

##########################################################################################
# utilities
##########################################################################################
_max_sets_radius(εs::Vector{<:Real}) = εs[1] # assumes sorted!
_max_sets_radius(εs::Vector{<:AbstractVector}) = maximum(εs[1]) # assumes sorted!
_default_threshold_distance(εs, m = 4) = m*_max_sets_radius(εs)

##########################################################################################
# old code that is left here for reference sake only (to check back on Callbacks)
##########################################################################################
#=

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

function mean_return_times(ds::ContinuousDynamicalSystem, u0, εs, T;
        i=10, dmin=_default_threshold_distance(εs), diffeq = NamedTuple(), kwargs...
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

=#