# Discrete dynamical systems (maps) implementation. Ultra fast, ultra clean.
import ProgressMeter
function exit_entry_times(integ::MDI, u₀, εs, T; show_progress = false, kwargs...)
    E = length(εs)
    prev_outside = fill(false, E)      # `true` if outside the set. Previous step.
    curr_outside = copy(prev_outside)  # `true` if outside the set. Current step.
    exits   = [Int[] for _ in 1:E]
    entries = [Int[] for _ in 1:E]
    prog = ProgressMeter.Progress(T; desc="Exit-entry times:", enabled=show_progress)
    t0 = integ.t
    while (integ.t - t0) < T
        step!(integ)
        # here i gives the index of the largest ε-set that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(get_state(integ), u₀, εs, E)
        curr_outside[i:end] .= true
        curr_outside[1:(i - 1)] .= false
        update_exit_times!(exits, i, prev_outside, curr_outside, integ.t)
        update_entry_times!(entries, i, prev_outside, curr_outside, integ.t)
        prev_outside .= curr_outside
        ProgressMeter.update!(prog, integ.t - t0)
    end
    return exits, entries
end

# These two functions add an entry to the exit or entry times,
# depending on whether the current and previous "outside" statuses
function update_exit_times!(exits, i, prev_outside, curr_outside, t::Int)
    @inbounds for j in i:length(prev_outside)
        curr_outside[j] && !prev_outside[j] && push!(exits[j], t)
    end
end
function update_entry_times!(entries, i, prev_outside, curr_outside, t::Int)
    @inbounds for j in 1:i-1
        prev_outside[j] && !curr_outside[j] && push!(entries[j], t)
    end
end



function first_return_time(integ::MDI, u₀, ε, T; show_progress = false, kwargs...)
    isout = false
    prog = ProgressMeter.Progress(T; desc="Exit-entry times:", enabled=show_progress)
    t0 = integ.t
    while !isout
        step!(integ)
        isout = isoutside(get_state(integ), u₀, ε)
    end
    while (integ.t - t0) < T
        step!(integ)
        ProgressMeter.update!(prog, integ.t - t0)
        isout = isoutside(get_state(integ), u₀, ε)
        if !isout
            ProgressMeter.finish!(prog)
            return integ.t - t0
        end
    end
    return NaN # in case it didn't return up to the max time
end
