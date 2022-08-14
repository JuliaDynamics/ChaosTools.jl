# The recurrences based method is rather flexible because it works
# in two independent steps: it first finds attractors and then matches them.

function basins_fractions_continuation(mapper, ps, pidx, ics::Function;
        seeds_per_attractor = 5, samples_per_parameter = 100, threshold = Inf,
    )
    rng = Random.Xoshiro()
    # At each parmaeter `p`, a dictionary mapping attractor ID to fraction is created.
    fractions_curves = Dict{Int8, Float64}[]
    # first parameter is run in isolation, as it has no prior to seed from
    set_parameter!(mapper.integ, pidx, ps[1])
    fs = basins_fractions(mapper, ics; show_progress = false, N = samples_per_parameter)
    push!(fractions_curves, fs)
    prev_attractors = deepcopy(mapper.bsn_nfo.attractors)

    for p in ps[2:end]
        set_parameter!(mapper.integ, pidx, p)
        overwrite_dict!(prev_attractors, mapper.bsn_nfo.attractors)
        reset!(mapper)
        # Seed initial conditions from previous attractors
        for (i, att) in prev_attractors
            for j in 1:seeds_per_attractor
                u0 = rand(rng, att.data)
                mapper(u0) # we don't care about return value here.
            end
        end
        # Now poerform basin fractions estimation as normal, utilizing found attractors
        fs = basins_fractions(mapper, ics; show_progress = false, N = samples_per_parameter)

        # Find new attractors
        unique_attractor_ids!(prev_attractors, mapper.bsn_nfo.attractors, 1.0)

        # But also correctly set new keys to new dictionary.
        push!(fractions_curves, fs)
    end
    unique_attractor_ids!(fractions_curves, 1.0)
    return fractions_curves
end

function overwrite_dict!(old::Dict, new::Dict)
    empty!(old)
    for (k, v) in new
        old[k] = v
    end
end

function reset!(mapper::AttractorsViaRecurrences)
    empty!(mapper.bsn_nfo.attractors)
    mapper.bsn_nfo.basins .= 0
    # TODO: Also set attractor ID to 0. How?
    return
end