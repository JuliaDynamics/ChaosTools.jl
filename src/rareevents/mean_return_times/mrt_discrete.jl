# Discrete dynamical systems (maps) implementation. Ultra fast, ultra clean.
function exit_entry_times(integ::MDI, u₀, εs, T)
    reinit!(integ, u₀)
    E = length(εs)
    pre_outside = fill(false, E)      # `true` if outside the set. Previous step.
    cur_outside = copy(pre_outside)   # `true` if outside the set. Current step.
    exits   = [Int[] for _ in 1:E]
    entries = [Int[] for _ in 1:E]

    while (integ.t - integ.t0) < T
        step!(integ)

        # here i gives the index of the largest ε-set that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(get_state(integ), u₀, εs, E)
        cur_outside[i:end] .= true
        cur_outside[1:(i - 1)] .= false

        update_exit_times!(exits, i, pre_outside, cur_outside, integ)
        update_entry_times!(entries, i, pre_outside, cur_outside, integ)
        pre_outside .= cur_outside
    end
    return exits, entries
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
#