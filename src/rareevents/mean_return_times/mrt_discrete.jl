function exit_entry_times(ds::DiscreteDynamicalSystem, u0, εs, T; diffeq = NamedTuple())
    check_εs_sorting(εs, length(u0))
    integ = integrator(ds, u0)
    exit_entry_times(integ, u0, εs, T)
end

function exit_entry_times(integ::MDI, u0, εs, T)
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
        cur_outside[1:i-1] .= false

        update_exit_times!(exits, i, pre_outside, cur_outside, integ)
        update_entry_times!(entries, i, pre_outside, cur_outside, integ)
        pre_outside .= cur_outside
    end
    return exits, entries
end

function first_outside_index(integ::MDI, u0, εs, E)::Int
    i = findfirst(e -> isoutside(integ.u, u0, e), εs)
    return isnothing(i) ? E+1 : i
end

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

function mean_return_times(ds::DiscreteDynamicalSystem, u0, εs, T; diffeq = NamedTuple())
    check_εs_sorting(εs, length(u0))
    integ = integrator(ds, u0)
    mean_return_times(integ, u0, εs, T)
end

function mean_return_times(integ::MDI, u0, εs, T)
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