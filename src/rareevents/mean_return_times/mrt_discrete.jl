# Discrete dynamical systems (maps) implementation. Ultra fast, ultra clean.
import ProgressMeter
function exit_entry_times(integ::MDI, u₀, εs, T; show_progress = false,
    internal_max_counter = Inf, kwargs...)
    E = length(εs)
    # TODO: Simply adjusting the initial value of `prev_outside` according to
    # arbitrary `u0` will allow this algorithm to work for any starting poing of integrator.
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
        i = first_outside_index(current_state(integ), u₀, εs, E)
        curr_outside[i:end] .= true
        curr_outside[1:(i - 1)] .= false
        update_exit_times!(exits, i, prev_outside, curr_outside, integ.t)
        update_entry_times!(entries, i, prev_outside, curr_outside, integ.t)
        prev_outside .= curr_outside
        ProgressMeter.update!(prog, integ.t - t0)
        if all(ex -> length(ex) ≥ internal_max_counter, exits) &&
            all(en -> length(en) ≥ internal_max_counter, entries)
            break # This clause exists only for `first_return_times` function.
        end
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
    # TODO: I'm lazy and I'm coding this by calling the normal algorithm.
    # A bit inneficient, maybe in the future someone can write a full version here...
    exits, entries = exit_entry_times(
        integ, u₀, εs, T; show_progress = false,
        internal_max_counter = 1
    )
    transits, returns = transit_return_times(exits, entries)
    rtimes = [r[1] for r in returns]
    return rtimes
end
