# Discrete dynamical systems (maps) implementation. Ultra fast, ultra clean.
function exit_entry_times(integ::MDI, u0, εs, T)
    reinit!(integ, u0)
    E = length(εs)
    pre_outside = fill(false, E)      # `true` if outside the set. Previous step.
    cur_outside = copy(pre_outside)   # `true` if outside the set. Current step.
    exits   = [Int[] for _ in 1:E]
    entries = [Int[] for _ in 1:E]

    while integ.t < T
        step!(integ)

        # here i gives the index of the largest ε-set that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(integ, u0, εs, E)
        cur_outside[i:end] .= true
        cur_outside[1:(i - 1)] .= false

        update_exit_times!(exits, i, pre_outside, cur_outside, integ)
        update_entry_times!(entries, i, pre_outside, cur_outside, integ)
        pre_outside .= cur_outside
    end
    return exits, entries
end

"Find the (index of the) outermost ε-ball the trajectory is not in."
function first_outside_index(integ::MDI, u0, εs, E)::Int
    i = findfirst(e -> isoutside(integ.u, u0, e), εs)
    return isnothing(i) ? E+1 : i
end

# These two functions add an entry to the exit or entry times,
# depending on whether the current and previous "outside" statuses
function update_exit_times!(exits, i, pre_outside, cur_outside, integ::MDI)
    @inbounds for j in i:length(pre_outside)
        cur_outside[j] && !pre_outside[j] && push!(exits[j], integ.t)
    end
end
function update_entry_times!(entries, i, pre_outside, cur_outside, integ::MDI)
    @inbounds for j in 1:i-1
        pre_outside[j] && !cur_outside[j] && push!(entries[j], integ.t)
    end
end

# Remember this is just a convenience function!
function mean_return_times(ds::DiscreteDynamicalSystem, u0, εs, T)
    exits, entries = exit_entry_times(ds, u0, εs, T)
    transits, returns = transit_return(exits, entries)
    mrt = mean.(returns)
    ret = length.(returns)
    return mrt, ret
end

# Old function. It optimized the return by not collecting the transit
# times, nor a vector of all returns, but rather an accumulated count.
# However, I believe the code simplicity is more important.
# Nevertheless, I am leavin the source code here in case someone needs to run
# insane amount of computations...
#=
function mean_return_times(ds::DiscreteDynamicalSystem, u0, εs, Τ)
    integ = integrator(ds, u0)
    E = length(εs)
    pre_outside = fill(false, E)      # `true` if outside the set. Previous step.
    cur_outside = copy(pre_outside)   # `true` if outside the set. Current step.
    exits, entries, counts, mrt = (zeros(Int, E) for _ in 1:4)

    while integ.t < T
        step!(integ)

        # here i gives the index of the largest ε-ball that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(integ, u0, εs, E)
        cur_outside[i:end] .= true
        cur_outside[1:i-1] .= false

        update_exit_times_single!(exits, i, pre_outside, cur_outside, integ.t)
        update_entries_and_returns!(
            mrt, counts, entries, exits, i, pre_outside, cur_outside, integ.t
        )
        pre_outside .= cur_outside
    end
    return mrt ./ counts, counts
end

function update_exit_times_single!(exits, i, pre_outside, cur_outside, t)
    @inbounds for j in i:length(pre_outside)
        cur_outside[j] && !pre_outside[j] && (exits[j] = t)
    end
end

function update_entries_and_returns!(
        mrt, counts, entries, exits, i, pre_outside, cur_outside, t
    )
    @inbounds for j in 1:i-1 # only when inside the set it makes sense to calculate returns
        if pre_outside[j] && !cur_outside[j] # we just entered the set
            entries[j] = t
            mrt[j] += entries[j] - exits[j]
            counts[j] += 1
        end
    end
end
=#