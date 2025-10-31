default_truncate_alg(tns::TensorNetworkState) = nothing

function ITensors.truncate(bpc::BeliefPropagationCache; bp_update_kwargs = default_bp_update_kwargs(bpc), maxdim::Integer, cutoff::Number = nothing)
    bpc = copy(bpc)
    s = siteinds(network(bpc))
    apply_kwargs = (; maxdim, cutoff)
    for e in edges(bpc)
        g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)]]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)]])
        apply_gate!(g1*g2, bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
        bpc = update(bpc; bp_update_kwargs...)
    end
    return bpc
end

function ITensors.truncate(bmps_cache::BoundaryMPSCache; maxdim::Integer, cutoff::Number = nothing)
    bmps_cache = copy(bmps_cache)
    s = siteinds(network(bmps_cache))
    apply_kwargs = (; maxdim, cutoff)

    ps = sort(partitionvertices(supergraph(bmps_cache)); by = v -> parent(v))
    for (i, p) in enumerate(ps)
        g = partition_graph(bmps_cache, p)
        leaves = leaf_vertices(g)
        seq = a_star(g, last(leaves), first(leaves))
        !isempty(seq) && update_partition!(bmps_cache, seq)
        for e in reverse.(reverse(seq))
            g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)]]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)]])
            envs = incoming_messages(bmps_cache, [src(e), dst(e)])
            ρv1, ρv2  = ITensorNetworks.full_update_bp(g1*g2, tensornetwork(network(bmps_cache)), [src(e), dst(e)]; envs, apply_kwargs...)
            setindex_preserve_graph!(bmps_cache, normalize(ρv1), src(e))
            setindex_preserve_graph!(bmps_cache, normalize(ρv2), dst(e))
            update_partition!(bmps_cache, [e])
        end

        if i != length(ps)
            bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = [PartitionEdge(parent(ps[i]) => parent(ps[i+1]))], maxiter = 1)
        end
    end

    return bmps_cache
end

function ITensors.truncate(alg::Algorithm"bp", tns::TensorNetworkState; kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache)
    bp_cache = truncate(bp_cache; kwargs...)
    return network(bp_cache)
end

function ITensors.truncate(alg::Algorithm"boundarymps", tns::TensorNetworkState; mps_bond_dimension::Integer, kwargs...)
    bmps_cache = BoundaryMPSCache(tns, mps_bond_dimension; partition_by = "row")
    leaves = leaf_vertices(partitions_graph(supergraph(bmps_cache)))
    seq = PartitionEdge.(a_star(partitions_graph(supergraph(bmps_cache)), last(leaves), first(leaves)))
    bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1)
    bmps_cache = truncate(bmps_cache; kwargs...)

    tns = network(bmps_cache)

    bmps_cache = BoundaryMPSCache(tns, mps_bond_dimension; partition_by = "col")
    leaves = leaf_vertices(partitions_graph(supergraph(bmps_cache)))
    seq = PartitionEdge.(a_star(partitions_graph(supergraph(bmps_cache)), last(leaves), first(leaves)))
    bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1)
    bmps_cache = truncate(bmps_cache; kwargs...)

    return network(bmps_cache)
end

function ITensors.truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    algorithm_check(tns, "truncate", alg)
    return truncate(Algorithm(alg), tns; kwargs...)
end

