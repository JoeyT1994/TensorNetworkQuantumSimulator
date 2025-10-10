default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(any) = error("You must specify a contraction algorithm.")

"""
    ITensorNetworks.expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork, observables::Vector{<:Tuple}, contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy()))

Function for computing expectation values for any vector of pauli strings via exact contraction.
This will be infeasible for larger networks with high bond dimension.
"""
function expect(
    alg::Algorithm"exact",
    ψ::TensorNetworkState,
    observables::Vector{<:Tuple};
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy())
)
    ITensors.disable_warn_order()

    denom = norm_sqr(alg, ψ; contraction_sequence_kwargs)
    out = []
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs)
        if iszero(coeff)
            push!(out, 0)
            continue
        end
        op_string_f = v -> v ∈ vs ? op_strings[findfirst(x -> x == v, vs)] : "I" 
        ψOψ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
        numer_seq = contraction_sequence(ψOψ_tensors; contraction_sequence_kwargs...)
        numer = contract(ψOψ_tensors; sequence=numer_seq)[]
        push!(out, numer / denom)
    end
    return out
end

function expect(alg::Algorithm"exact",
    ψ::TensorNetworkState,
    observable::Tuple;
    kwargs...
)
    return only(expect(alg, ψ, [observable]; kwargs...))
end

function expect(ψ::Union{TensorNetworkState, BeliefPropagationCache}, observable; alg::String = default_alg(ψ), kwargs...)
    return expect(Algorithm(alg), ψ, observable; kwargs...)
end

function expect(
    alg::Algorithm"bp",
    ψ::BeliefPropagationCache,
    obs::Tuple
)
    op_strings, obs_vs, coeff = collectobservable(obs)
    iszero(coeff) && return 0

    #For boundary MPS, must stay in partition
    steiner_vs = length(obs_vs) == 1 ? obs_vs : collect(vertices(steiner_tree(network(ψ), obs_vs)))
    op_string_f = v -> v ∈ obs_vs ? op_strings[findfirst(x->x == v, obs_vs)] : "I"

    incoming_ms = incoming_messages(ψ, steiner_vs)
    ψIψ_tensors = ITensor[norm_factors(network(ψ), steiner_vs); incoming_ms]
    denom_seq = contraction_sequence(ψIψ_tensors; alg = "optimal")
    denom = contract(ψIψ_tensors; sequence=denom_seq)[]

    ψOψ_tensors = ITensor[norm_factors(network(ψ), steiner_vs; op_strings = op_string_f); incoming_ms]
    numer_seq = contraction_sequence(ψOψ_tensors; alg = "optimal")
    numer = contract(ψOψ_tensors; sequence=numer_seq)[]

    return coeff * numer/ denom
end

function expect(
    alg::Algorithm"bp",
    ψ::TensorNetworkState,
    observable;
    bp_update_kwargs = default_bp_update_kwargs(ψ),
    kwargs...,
)

    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

    return expect(alg, ψ_bpc, observable; kwargs...)
end

function expect(
    alg::Algorithm,
    bp_cache::AbstractBeliefPropagationCache,
    observables::Vector{<:Tuple};
    kwargs...,
)
    return map(obs -> expect(alg, bp_cache, obs; kwargs...), observables)
end

#Process an observable into more readable form
function collectobservable(obs::Tuple)
    # unpack
    op = obs[1]
    qinds = obs[2]
    coeff = length(obs) == 2 ? 1 : last(obs)

    @assert !(op == "" && isempty(qinds))

    op_vec = [string(o) for o in op]
    qinds_vec = _tovec(qinds)
    return op_vec, qinds_vec, coeff
end

_tovec(qinds) = vec(collect(qinds))
_tovec(qinds::NamedEdge) = [qinds.src, qinds.dst]
