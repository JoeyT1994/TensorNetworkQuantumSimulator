default_expect_alg() = "bp"

function ITensorNetworks.expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork,
    observables::Vector{<:Tuple},
    contraction_sequence_kwargs = (; alg="einexpr", optimizer=Greedy()))

    s = siteinds(ψ)
    ψIψ = QuadraticFormNetwork(ψ) 

    out = []
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs)
        iszero(coeff) && return 0.0
        ψOψ = copy(ψIψ)
        for (op_string, v) in zip(op_strings, vs)
        ψOψ[(v, "operator")] = ITensors.op(op_string, s[v])
        end

        numer_seq = contraction_sequence(ψOψ; contraction_sequence_kwargs...)
        denom_seq = contraction_sequence(ψIψ; contraction_sequence_kwargs...)
        numer, denom = contract(ψOψ; sequence = numer_seq)[], contract(ψIψ; sequence = denom_seq)[]
        push!(out, numer / denom)
    end
    return out
end
  
function ITensorNetworks.expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork,
    obs::Tuple, kwargs...)
    return only(expect(alg, ψ, [obs]; kwargs...))
end
  
function ITensorNetworks.expect(
    alg::Algorithm,
    ψ::AbstractITensorNetwork,
    observables::Vector{<:Tuple};
    (cache!)=nothing,
    update_cache=isnothing(cache!),
    cache_update_kwargs=get_global_cache_update_kwargs(alg),
    cache_construction_kwargs=default_cache_construction_kwargs(alg, QuadraticFormNetwork(ψ)),
    kwargs...,
    )
    ψIψ = QuadraticFormNetwork(ψ)
    if isnothing(cache!)
        cache! = Ref(cache(alg, ψIψ; cache_construction_kwargs...))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    return expect(cache![], observables; alg, kwargs...)
end
  
function ITensorNetworks.expect(alg::Algorithm,
    ψ::AbstractITensorNetwork,
    observable::Tuple;
    kwargs...)
    return only(expect(alg, ψ, [observable]; kwargs...))
end
  
"""
    expect(ψ::AbstractITensorNetwork, obs::Tuple; kwargs...)

Calculate the expectation value of an `ITensorNetwork` `ψ` with an observable `obs` using the desired algorithm.
"""
function ITensorNetworks.expect(
    ψ::AbstractITensorNetwork,
    args...;
    alg = default_expect_alg(),
    kwargs...,
    )
    return expect(Algorithm(alg), ψ, args...; kwargs...)
end
  
"""
    expect(ψIψ::AbstractBeliefPropagationCache, obs::Tuple; kwargs...)

Foundational expectation function for a given (norm) cache network with an observable. 
This can be a `BeliefPropagationCache` or a `BoundaryMPSCache`.
Valid observables are tuples of the form `(op, qinds)` or `(op, qinds, coeff)`, 
where `op` is a string or vector of strings, `qinds` is a vector of indices, and `coeff` is a coefficient (default 1.0).
"""
function ITensorNetworks.expect(
    ψIψ::AbstractBeliefPropagationCache,
    obs::Tuple;
    update_cache = false,
    kwargs...,
    )

    op_strings, vs, coeff = collectobservable(obs)
    iszero(coeff) && return 0.0

    ψOψ = insert_observable(ψIψ, obs)

    numerator = region_scalar(ψOψ, [(v, "ket") for v  in vs])
    denominator = region_scalar(ψIψ, [(v, "ket") for v  in vs])

    return coeff * numerator / denominator
end
  
function ITensorNetworks.expect(ψIψ::AbstractBeliefPropagationCache, observables::Vector{<:Tuple}; kwargs...)
    return map(obs -> expect(ψIψ, obs; kwargs...), observables)
end

## utilites
function insert_observable(ψIψ::AbstractBeliefPropagationCache, obs)
    op_strings, verts, _ = collectobservable(obs)
  
    ψIψ_tn = tensornetwork(ψIψ)
    ψIψ_vs = [ψIψ_tn[operator_vertex(ψIψ_tn, v)] for v in verts]
    sinds = [commonind(ψIψ_tn[ket_vertex(ψIψ_tn, v)], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    operators = [ITensors.op(op_strings[i], sinds[i]) for i in eachindex(op_strings)]
  
    ψOψ = update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))
    return ψOψ
end

function collectobservable(obs::Tuple)
    # unpack
    op = obs[1]
    qinds = obs[2]
    coeff = length(obs) == 2 ? 1.0 : last(obs)
  
    @assert !(op == "" && isempty(qinds))
  
    op_vec = [string(o) for o in op]
    qinds_vec = _tovec(qinds)
    return op_vec, qinds_vec, coeff
  end

_tovec(qinds) = vec(collect(qinds))
_tovec(qinds::NamedEdge) = [qinds.src, qinds.dst]


