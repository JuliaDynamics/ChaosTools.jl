export RecurrencesSeedingContinuation, basins_fractions_continuation
# The recurrences based method is rather flexible because it works
# in two independent steprange: it first finds attractors and then matches them.
struct RecurrencesSeedingContinuation{A, M, S, E}
    mapper::A
    metric::M
    threshold::Float64
    seeds_from_attractor::S
    info_extraction::E
end


"""
    RecurrencesSeedingContinuation(mapper::AttractorsViaRecurrences; kwargs...)
A method for [`basins_fractions_continuation`](@ref).
It uses seeding of previous attractors to find new ones, which is the main performance
bottleneck. Will write more once we have the paper going.

## Keyword Arguments
- `metric, threshold`: Given to [`match_attractor_ids!`](@ref) which is the function
  used to match attractors between each parameter slice.
- `info_extraction = identity`: A function that takes as an input an attractor (`Dataset`)
  and outputs whatever information should be stored. It is used to return the
  `attractors_info` in [`basins_fractions_continuation`](@ref).
- `seeds_from_attractor`: A function that takes as an input an attractor and returns
  an iterator of initial conditions to be seeded from the attractor for the next
  parameter slice. By default, we sample some points from existing attractors according
  to how many points the attractors themselves contain. A maximum of `10` seeds is done
  per attractor.
"""
function RecurrencesSeedingContinuation(
        mapper::AttractorsViaRecurrences; metric = Euclidean(),
        threshold = Inf, seeds_from_attractor = _default_seeding_process,
        info_extraction = identity
    )
    return RecurrencesSeedingContinuation(
        mapper, metric, threshold, seeds_from_attractor, info_extraction
    )
end

function _default_seeding_process(attractor::AbstractDataset)
    max_possible_seeds = 10
    seeds = round(Int, log(10, length(attractor)))
    seeds = clamp(seeds, 1, max_possible_seeds)
    return (rand(attractor.data) for _ in 1:seeds)
end

function basins_fractions_continuation(
        continuation::RecurrencesSeedingContinuation, prange, pidx, ics::Function;
        samples_per_parameter = 100, show_progress = false,
    )
    show_progress && @info "Starting basins fraction continuation."
    show_progress && @info "p = $(prange[1])"
    (; mapper, metric, threshold) = continuation
    # first parameter is run in isolation, as it has no prior to seed from
    set_parameter!(mapper.integ, pidx, prange[1])
    fs = basins_fractions(mapper, ics; show_progress, N = samples_per_parameter)
    # At each parmaeter `p`, a dictionary mapping attractor ID to fraction is created.
    fractions_curves = [fs]
    # Furthermore some info about the attractors is stored and returned
    prev_attractors = deepcopy(mapper.bsn_nfo.attractors)
    get_info = attractors -> Dict(k => continuation.info_extraction(att) for (k, att) in attractors)
    info = get_info(prev_attractors)
    attractors_info = [info]

    for p in prange[2:end]
        # TODO: Make this use ProgressMeter.jl
        show_progress && @show p
        set_parameter!(mapper.integ, pidx, p)
        reset!(mapper)
        # Seed initial conditions from previous attractors
        for att in values(prev_attractors)
            for u0 in continuation.seeds_from_attractor(att)
                # We map the initial condition to an attractor, but we don't care
                # about which attractor we go to. This is just so that the internal
                # array of `AttractorsViaRecurrences` registers the attractors
                mapper(u0; show_progress)
            end
        end
        # Now perform basin fractions estimation as normal, utilizing found attractors
        fs = basins_fractions(mapper, ics; show_progress = false, N = samples_per_parameter)
        current_attractors = mapper.bsn_nfo.attractors
        # Match with previous attractors before storing anything!
        @show rmap = match_attractor_ids!(current_attractors, prev_attractors; metric, threshold)

        # Then do the remaining setup for storing and next step
        swap_dict_keys!(fs, rmap)
        overwrite_dict!(prev_attractors, current_attractors)
        push!(fractions_curves, fs)
        push!(attractors_info, get_info(prev_attractors))
    end

    srmap = renumber_keys_sequentially!(attractors_info, fractions_curves)
    @show srmap
    return fractions_curves, attractors_info
end

function reset!(mapper::AttractorsViaRecurrences)
    empty!(mapper.bsn_nfo.attractors)
    if mapper.bsn_nfo.basins isa Array
        mapper.bsn_nfo.basins .= 0
    else
        empty!(mapper.bsn_nfo.basins)
    end
    mapper.bsn_nfo.state = :att_search
    mapper.bsn_nfo.consecutive_match = 0
    mapper.bsn_nfo.consecutive_lost = 0
    mapper.bsn_nfo.prev_label = 0
    # notice that we do not reset the following:
    # mapper.bsn_nfo.current_att_label = 2
    # mapper.bsn_nfo.visited_cell = 4
    # because we want the next attractor to be labelled differently in case
    # it doesn't actually match to any of the new ones
    return
end

function renumber_keys_sequentially!(att_info, frac_curves)
    # First collect all unique keys (hence using `Set`)
    keys = Set{Int16}()
    for af in att_info
        for k in af
            push!(keys,k[1])
        end
    end

    # Now set up a replacement map
    rmap = Dict()
    for (j, ke) in enumerate(keys)
       push!(rmap, ke => j)
    end

    for fs in frac_curves
        swap_dict_keys!(fs, rmap)
    end
    for af in att_info
        swap_dict_keys!(af, rmap)
    end
    return rmap
end