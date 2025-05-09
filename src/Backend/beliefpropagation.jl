const _default_bp_update_maxiter = 25
const _default_bp_update_tol = 1e-10

## Frontend functions

function default_posdef_bp_update_kwargs()
    message_update_function = ms -> make_eigs_real.(default_message_update(ms))
    return (; maxiter = _default_bp_update_maxiter, tol = _default_bp_update_tol, message_update_kwargs = (; message_update_function))
end

function default_nonposdef_bp_update_kwargs()
    message_update_function = ms -> default_message_update(ms)
    return (; maxiter = _default_bp_update_maxiter, tol = _default_bp_update_tol, message_update_kwargs = (; message_update_function))
end

function default_bp_update_kwargs(bp_cache::BeliefPropagationCache)
    return default_nonposdef_bp_update_kwargs()
end

function updatecache(bp_cache::BeliefPropagationCache; kwargs...)
    return update(bp_cache; kwargs...)
end

function build_bp_cache(
    ψ::AbstractITensorNetwork,
    args...;
    update_cache = true,
    cache_update_kwargs = default_posdef_bp_update_kwargs(),
)
    bp_cache = BeliefPropagationCache(QuadraticFormNetwork(ψ), args...)
    # TODO: QuadraticFormNetwork() builds ψIψ network, but for Pauli picture `norm_sqr_network()` is enough
    # https://github.com/ITensor/ITensorNetworks.jl/blob/main/test/test_belief_propagation.jl line 49 to construct the cache without the identities.
    if update_cache
        bp_cache = updatecache(bp_cache; cache_update_kwargs...)
    end
    return bp_cache
end

function is_flat(bpc::BeliefPropagationCache)
    pg = partitioned_tensornetwork(bpc)
    return all([length(vertices(pg, pv)) == 1 for pv in partitionvertices(pg)])
end

function symmetric_gauge(ψ::AbstractITensorNetwork; cache_update_kwargs = default_posdef_bp_update_kwargs(), kwargs...)
    ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs, kwargs...)
    cache_ref = Ref{BeliefPropagationCache}()
    ψ_symm = ITensorNetwork(ψ_vidal; (cache!) = cache_ref)
    bp_cache = cache_ref[]
    return ψ_symm, bp_cache
end

function LinearAlgebra.normalize(
    ψ::ITensorNetwork,
    ψψ_bpc::BeliefPropagationCache;
    cache_update_kwargs = default_posdef_bp_update_kwargs(),
    update_cache = false,
)
    ψψ_bpc_ref = Ref(copy(ψψ_bpc))
    ψ = normalize(ψ; alg = "bp", cache! = ψψ_bpc_ref, cache_update_kwargs, update_cache)

    return ψ, ψψ_bpc_ref[]
end

function ITensors.scalar(
    bp_cache::AbstractBeliefPropagationCache,
    args...;
    alg = "bp",
    kwargs...,
)
    return scalar(Algorithm(alg), bp_cache, args...; kwargs...)
end

function ITensors.scalar(alg::Algorithm"bp", bp_cache::AbstractBeliefPropagationCache)
    return scalar(bp_cache)
end

function ITensorNetworks.region_scalar(bpc::BeliefPropagationCache, verts::Vector)
    partitions = partitionvertices(bpc, verts)
    length(partitions) == 1 && return region_scalar(bpc, only(partitions))
    if length(partitions) == 2
        p1, p2 = first(partitions), last(partitions)
        if parent(p1) ∉ neighbors(partitioned_graph(bpc), parent(p2))
            error(
                "Only contractions involving neighboring partitions are currently supported",
            )
        end
        ms = incoming_messages(bpc, partitions)
        local_tensors = factors(bpc, partitions)
        ts = [ms; local_tensors]
        seq = contraction_sequence(ts; alg = "optimal")
        return contract(ts; sequence = seq)[]
    end
    error("Contractions involving more than 2 partitions not currently supported")
    return nothing
end


"""Bipartite entanglement entropy, estimated as the spectrum of the bond tensor on the bipartition edge."""
function entanglement(
    ψ::ITensorNetwork,
    e::NamedEdge;
    (cache!) = nothing,
    cache_update_kwargs = default_posdef_bp_update_kwargs(),
)
    cache = isnothing(cache!) ? build_bp_cache(ψ; cache_update_kwargs) : cache![]
    ψ_vidal = VidalITensorNetwork(ψ; cache)
    bt = ITensorNetworks.bond_tensor(ψ_vidal, e)
    ee = 0
    for d in diag(bt)
        ee -= abs(d) >= eps(eltype(bt)) ? d * d * log2(d * d) : 0
    end
    return abs(ee)
end


function make_eigs_real(A::ITensor)
    return map_eigvals(x -> real(x), A, first(inds(A)), last(inds(A)); ishermitian = true)
end

function make_eigs_positive(A::ITensor, tol::Real = 1e-14)
    return map_eigvals(
        x -> max(x, tol),
        A,
        first(inds(A)),
        last(inds(A));
        ishermitian = true,
    )
end

function delete_message!(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge)
    return delete_messages!(bpc, [pe])
end

function delete_messages!(bpc::AbstractBeliefPropagationCache, pes::Vector)
    ms = messages(bpc)
    for pe in pes
        haskey(ms, pe) && delete!(ms, pe)
    end
    return bpc
end
