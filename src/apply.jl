### Questions:
# BP update control via "update_every" parameter?
# What about normalization?
# How do we handle the BP cache? Build a new object that has both?

const _default_apply_kwargs =
    (maxdim = Inf, cutoff = 1e-10, normalize = true, use_relative_cutoff = true)

# import ITensorNetworks.apply for less clutter down below
# import ITensors.apply
# import ITensorNetworks.apply

# apply a circuit to a tensor network without BP cache being passed
function ITensors.apply(
    circuit::AbstractVector,
    ψ::ITensorNetwork;
    bp_update_kwargs = get_global_bp_update_kwargs(),
    kwargs...,
)
    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ψ, ψψ, truncation_errors = apply(circuit, ψ, ψψ; kwargs...)
    # given that people used this function, we assume they don't want the cache
    return ψ, truncation_errors
end

# convert the circuit to a vector of itensors
function ITensors.apply(
    circuit::AbstractVector,
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    kwargs...,
)
    circuit = toitensor(circuit, siteinds(ψ))
    return apply(circuit, ψ, ψψ; kwargs...)
end

# the main simulation function
function ITensors.apply(
    circuit::AbstractVector{<:ITensor},
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = get_global_bp_update_kwargs(),
    update_every = 1,
    verbose = false,
)

    # merge all the kwargs with the defaults 
    apply_kwargs = merge(_default_apply_kwargs, apply_kwargs)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_vertices = Set{Index{Int64}}()
    counter = 0
    truncation_errors = zeros((length(circuit)))

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)

        # actually apply the gate
        t = @timed ψ, ψψ, truncation_errors[ii] = apply(gate, ψ, ψψ; apply_kwargs)

        if verbose
            println(
                "Gate $ii:    Simulation time: $(t.time) secs,    Max χ: $(maxlinkdim(ψ)),     Error: $(truncation_errors[ii])",
            )
        end

        # TODO: the update should actual happen before the gate is applied
        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        counter, affected_vertices =
            _check_and_update_counter(counter, affected_vertices, gate)

        # update the BP cache
        if (counter > 0) && (counter % update_every == 0)
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψψ = updatecache(ψψ; bp_update_kwargs...)

            if verbose
                println("Done in $(t.time) secs")
            end

            # reset counter and affected_vertices
            counter = 0
            empty!(affected_vertices)
        end

    end
    return ψ, ψψ, truncation_errors
end

# for convenience an apply function for a single gate
function ITensors.apply(
    gate::Tuple,
    ψ::ITensorNetwork;
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = get_global_bp_update_kwargs(),
)
    ψ, ψψ, truncation_error =
        apply(gate, ψ, build_bp_cache(ψ; bp_update_kwargs...); apply_kwargs)
    # because the cache is not passed, we return the state only
    return ψ, truncation_error
end

# gate apply function for tuple gates. The gate gets converted to an ITensor first.
function ITensors.apply(
    gate::Tuple,
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    apply_kwargs = _default_apply_kwargs,
)
    return apply(
        toitensor(gate, siteinds(ψ)),
        ψ,
        ψψ;
        reset_all_messages = false,
        apply_kwargs,
    )
end


function ITensors.apply(
    gate::ITensor,
    ψ::AbstractITensorNetwork,
    ψψ::BeliefPropagationCache;
    reset_all_messages = false,
    apply_kwargs = _default_apply_kwargs,
)
    # TODO: document each line

    ψψ = copy(ψψ)
    ψ = copy(ψ)
    vs = neighbor_vertices(ψ, gate)
    envs = incoming_messages(ψψ, PartitionVertex.(vs))

    err = 0.0
    s_values = ITensor(1.0)
    function callback(; singular_values, truncation_error)
        err = truncation_error
        s_values = singular_values
        return nothing
    end

    # this is the only call to a lower-level apply that we currently do.
    ψ = noprime(ITensorNetworks.apply(gate, ψ; envs, callback, apply_kwargs...))

    ψdag = prime(dag(ψ))
    if length(vs) == 2
        v1, v2 = vs
        pe = partitionedge(ψψ, (v1, "bra") => (v2, "bra"))
        mts = messages(ψψ)
        ind2 = commonind(s_values, ψ[v1])
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        if !reset_all_messages
            set!(mts, pe, dag.(ITensor[s_values]))
            set!(mts, reverse(pe), ITensor[s_values])
        else
            ψψ = BeliefPropagationCache(partitioned_tensornetwork(ψψ))
        end
    end
    for v in vs
        ψψ = update_factor(ψψ, (v, "ket"), ψ[v])
        ψψ = update_factor(ψψ, (v, "bra"), ψdag[v])
    end
    return ψ, ψψ, err
end


function _check_and_update_counter(counter::Integer, affected_vertices::Set, gate::ITensor)
    indices = inds(gate)

    # there is no scenario where we want to increment the counter for a single-qubit gate
    # they also don't count as affected vertices
    if length(indices) == 4
        # check if any of the qinds are in the affected_vertices
        # this increments the counter to update the cache
        if any(ind in affected_vertices for ind in indices)
            counter += 1
        end

        # if any of them is already contained, the set will take care of that
        affected_vertices = union(affected_vertices, Set(indices))
    end

    return counter, affected_vertices
end
