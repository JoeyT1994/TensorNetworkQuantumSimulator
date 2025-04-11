function sim_edgeinduced_subgraph(bpc::BeliefPropagationCache, eg)
    bpc = copy(bpc)
    pvs = PartitionVertex.(collect(vertices(eg)))
    pes = unique(reduce(vcat, [boundary_partitionedges(bpc, [pv]; dir = :out) for pv in pvs]))
    updated_pes = PartitionEdge[]
    antiprojectors = ITensor[]
    for pe in pes
        if reverse(pe) ∉ updated_pes
            mer =only(message(bpc, reverse(pe)))
            linds = inds(mer)
            linds_sim = sim.(linds)
            mer = replaceinds(mer, linds, linds_sim)
            ms = messages(bpc)
            set!(ms, reverse(pe), ITensor[mer])
            verts = vertices(bpc, src(pe))
            for v in verts
                t = only(factors(bpc, [v]))
                t_inds = filter(i -> i ∈ linds, inds(t))
                if !isempty(t_inds)
                    t_ind = only(t_inds)
                    t_ind_pos = findfirst(x -> x == t_ind, linds)
                    t = replaceind(t, t_ind, linds_sim[t_ind_pos])
                    bpc = update_factor(bpc, v, t)
                end
            end
            push!(updated_pes, pe)

            if pe ∈ PartitionEdge.(edges(eg)) || reverse(pe) ∈ PartitionEdge.(edges(eg))
                row_inds, col_inds = linds, linds_sim
                row_combiner, col_combiner = combiner(row_inds), combiner(col_inds)
                ap = delta(combinedind(row_combiner), combinedind(col_combiner)) * row_combiner * col_combiner
                ap =  ap - only(message(bpc, pe)) * mer
                push!(antiprojectors, ap)
            end
        end
    end
    return bpc, antiprojectors
end

function ITensorNetworks.boundary_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    pvs = unique(vcat(src.(pes), dst.(pes)))
    bpes = PartitionEdge[]
    for pv in pvs
        incoming_es = boundary_partitionedges(bpc, pv; dir = :in)
        incoming_es = filter(e -> e ∉ pes && reverse(e) ∉ pes, incoming_es)
        append!(bpes, incoming_es)
    end
    return bpes
end

#Return weight of term with anti projectors on all edges in the edge induced subgraph and messages everywhere else
function weight(bpc::BeliefPropagationCache, eg)
    pvs = PartitionVertex.(collect(vertices(eg)))
    pes = PartitionEdge.(collect(edges(eg)))
    bpc, antiprojectors = sim_edgeinduced_subgraph(bpc, eg)
    incoming_ms = ITensor[only(message(bpc, pe)) for pe in boundary_partitionedges(bpc, pes)]
    local_tensors = factors(bpc, vertices(bpc, pvs))
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = contraction_sequence(ts; alg="einexpr", optimizer=Greedy())
    return contract(ts;  sequence = seq)[]
end

function weights(bpc, egs, args...)
    return [weight(bpc, eg, args...) for eg in egs]
end