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



function first_return_times(integ::MDI, u₀, εs, T; show_progress = false, kwargs...)
    prog = ProgressMeter.Progress(T; desc="First return times:", enabled=show_progress)
    isout = false
    maxε = εs[1]
    rtimes = zeros(Int, length(εs))
    while !isout
        step!(integ)
        isout = isoutside(get_state(integ), u₀, maxε)
    end
    t0 = integ.t # so to not count the exit step as well.

    while (integ.t - t0) < T
        step!(integ)
        ProgressMeter.update!(prog, integ.t - t0)
        isout = isoutside(get_state(integ), u₀, maxε)
        while !isout
            rtimes[j] = integ.t - t0
            j += 1 # encoded the first return, now we continue into the deeper level
            if j > length(εs)
                @goto finish # goes to the `@label` below
            end
            maxε = εs[j]
            # We check the next set in the same loop in case we entered deeper than 1 level
            isout = isoutside(get_state(integ), u₀, maxε)
        end
    end
    @label finish
    ProgressMeter.finish!(prog)
    return rtimes
end
