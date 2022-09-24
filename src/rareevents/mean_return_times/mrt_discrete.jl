# Discrete dynamical systems (maps) implementation. Ultra fast, ultra clean.
function exit_entry_times(integ::MDI, u₀, εs, T; kwargs...)
    E = length(εs)
    prev_outside = fill(false, E)      # `true` if outside the set. Previous step.
    curr_outside = copy(prev_outside)  # `true` if outside the set. Current step.
    exits   = [Int[] for _ in 1:E]
    entries = [Int[] for _ in 1:E]

    while (integ.t - integ.t0) < T
        step!(integ)
        # here i gives the index of the largest ε-set that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(get_state(integ), u₀, εs, E)
        curr_outside[i:end] .= true
        curr_outside[1:(i - 1)] .= false
        update_exit_times!(exits, i, prev_outside, curr_outside, integ)
        update_entry_times!(entries, i, prev_outside, curr_outside, integ)
        prev_outside .= curr_outside
    end
    return exits, entries
end

# These two functions add an entry to the exit or entry times,
# depending on whether the current and previous "outside" statuses
function update_exit_times!(exits, i, prev_outside, curr_outside, integ::MDI)
    @inbounds for j in i:length(prev_outside)
        curr_outside[j] && !prev_outside[j] && push!(exits[j], integ.t)
    end
end
function update_entry_times!(entries, i, prev_outside, curr_outside, integ::MDI)
    @inbounds for j in 1:i-1
        prev_outside[j] && !curr_outside[j] && push!(entries[j], integ.t)
    end
end
