struct BoundaryMPSCache{BPC,PG} <: AbstractBeliefPropagationCache
    bp_cache::BPC
    partitionedplanargraph::PG
    maximum_virtual_dimension::Int64
end

## Utilities to globally set boundary MPS_update_kwargs 
const _default_boundarymps_update_alg = "orthogonal"
const _default_boundarymps_update_niters = 40
const _default_boundarymps_update_tolerance = 1e-12


# we make this a Dict that it can be pushed to with kwargs that we haven't thought of
# TODO: make this is not quite correct for boundary MPS
const _global_boundarymps_update_kwargs::Dict{Symbol,Any} = Dict(
    :alg => _default_boundarymps_update_alg,
    :message_update_kwargs => (;
        niters = _default_boundarymps_update_niters,
        tolerance = _default_boundarymps_update_tolerance,
    ),
)

function set_global_boundarymps_update_kwargs!(; kwargs...)
    for (arg, val) in kwargs
        _global_boundarymps_update_kwargs[arg] = val
    end
    return get_global_boundarymps_update_kwargs()
end

function get_global_boundarymps_update_kwargs()
    # return as a named tuple
    return (; _global_boundarymps_update_kwargs...)
end

function reset_global_boundarymps_update_kwargs!()
    empty!(_global_bp_update_kwargs)
    _global_boundarymps_update_kwargs[:alg] = _default_boundarymps_update_alg
    _global_boundarymps_update_kwargs[:message_update_kwargs] = (;
        niters = _default_boundarymps_update_niters,
        tolerance = _default_boundarymps_update_tolerance,
    )
    return get_global_boundarymps_update_kwargs()
end

## Frontend functions

function updatecache(bp_cache::BoundaryMPSCache, args...; boundarymps_update_kwargs...)
    # merge provided kwargs with the defaults
    boundarymps_update_kwargs =
        merge(get_global_boundarymps_update_kwargs(), boundarymps_update_kwargs)

    return update(bp_cache, args...; boundarymps_update_kwargs...)
end

function build_boundarymps_cache(
    ψ::AbstractITensorNetwork,
    message_rank::Int64;
    cache_construction_kwargs = (;),
    boundary_mps_kwargs...,
)
    # build the BP cache
    ψIψ = build_bp_cache(ψ; update_cache = false)

    # convert BP cache to boundary MPS cache, no further update needed
    return build_boundarymps_cache(
        ψIψ,
        message_rank;
        cache_construction_kwargs,
        boundary_mps_kwargs...,
    )
end

function build_boundarymps_cache(
    ψIψ::AbstractBeliefPropagationCache,
    message_rank::Int64;
    update_cache = true,
    cache_construction_kwargs = (;),
    boundary_mps_kwargs...,
)

    ψIψ = BoundaryMPSCache(ψIψ; message_rank, cache_construction_kwargs...)

    if update_cache
        # update the cache
        ψIψ = updatecache(ψIψ; boundary_mps_kwargs...)
    end

    return ψIψ
end

# a version for the inner product of two state networks
function build_boundarymps_cache(
    ψ::AbstractITensorNetwork,
    ϕ::AbstractITensorNetwork,
    message_rank::Int64;
    boundary_mps_kwargs...,
)

    ψϕ = build_bp_cache(ψ, ϕ; update_cache = false)

    # convert BP cache to boundary MPS cache, no further update needed
    return build_boundarymps_cache(ψϕ, message_rank; boundary_mps_kwargs...)
end



function build_boundarymps_cache(ψIψ::BoundaryMPSCache, args...; kwargs...)
    return ψIψ
end

## Backend functions
bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
maximum_virtual_dimension(bmpsc::BoundaryMPSCache) = bmpsc.maximum_virtual_dimension
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function ITensorNetworks.partitioned_tensornetwork(bmpsc::BoundaryMPSCache)
    return partitioned_tensornetwork(bp_cache(bmpsc))
end
ITensorNetworks.messages(bmpsc::BoundaryMPSCache) = messages(bp_cache(bmpsc))

ITensorNetworks.default_message_update_alg(bmpsc::BoundaryMPSCache) = "orthogonal"

function ITensorNetworks.default_bp_maxiter(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
)
    return default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
end
ITensorNetworks.default_bp_maxiter(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) =
    50
function ITensorNetworks.default_edge_sequence(alg::Algorithm, bmpsc::BoundaryMPSCache)
    return pair.(default_edge_sequence(ppg(bmpsc)))
end

function default_message_update_kwargs(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
    return (; niters = 50, tolerance = 1e-10)
end
default_boundarymps_message_rank(tn::AbstractITensorNetwork) = maxlinkdim(tn)^2
ITensorNetworks.partitions(bmpsc::BoundaryMPSCache) =
    parent.(collect(partitionvertices(ppg(bmpsc))))
ITensorNetworks.partitionpairs(bmpsc::BoundaryMPSCache) = pair.(partitionedges(ppg(bmpsc)))

function ITensorNetworks.cache(
    alg::Algorithm"boundarymps",
    tn;
    bp_cache_construction_kwargs = default_cache_construction_kwargs(Algorithm("bp"), tn),
    kwargs...,
)
    return BoundaryMPSCache(
        BeliefPropagationCache(tn; bp_cache_construction_kwargs...);
        kwargs...,
    )
end

function ITensorNetworks.default_cache_construction_kwargs(alg::Algorithm"boundarymps", tn)
    return (;
        bp_cache_construction_kwargs = default_cache_construction_kwargs(
            Algorithm("bp"),
            tn,
        )
    )
end

function ITensorNetworks.default_cache_update_kwargs(alg::Algorithm"boundarymps")
    return get_global_boundarymps_update_kwargs()
end

function Base.copy(bmpsc::BoundaryMPSCache)
    return BoundaryMPSCache(
        copy(bp_cache(bmpsc)),
        copy(ppg(bmpsc)),
        maximum_virtual_dimension(bmpsc),
    )
end

function ITensorNetworks.default_message(
    bmpsc::BoundaryMPSCache,
    pe::PartitionEdge;
    kwargs...,
)
    return default_message(bp_cache(bmpsc), pe::PartitionEdge; kwargs...)
end

#Get the dimension of the virtual index between the two message tensors on pe1 and pe2
function virtual_index_dimension(
    bmpsc::BoundaryMPSCache,
    pe1::PartitionEdge,
    pe2::PartitionEdge,
)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe1))

    if findfirst(x -> x == pe1, pes) > findfirst(x -> x == pe2, pes)
        lower_pe, upper_pe = pe2, pe1
    else
        lower_pe, upper_pe = pe1, pe2
    end
    inds_above = reduce(vcat, linkinds.((bmpsc,), partitionedges_above(bmpsc, lower_pe)))
    inds_below = reduce(vcat, linkinds.((bmpsc,), partitionedges_below(bmpsc, upper_pe)))

    return Int(minimum((
        prod(Float64.(dim.(inds_above))),
        prod(Float64.(dim.(inds_below))),
        Float64(maximum_virtual_dimension(bmpsc)),
    )))
end

#Vertices of the planargraph
function planargraph_vertices(bmpsc::BoundaryMPSCache, partition)
    return vertices(ppg(bmpsc), PartitionVertex(partition))
end

#Get partition(s) of vertices of the planargraph
function planargraph_partitions(bmpsc::BoundaryMPSCache, vertices::Vector)
    return parent.(partitionvertices(ppg(bmpsc), vertices))
end

function planargraph_partition(bmpsc::BoundaryMPSCache, vertex)
    return only(planargraph_partitions(bmpsc, [vertex]))
end

#Get interpartition pairs of partition edges in the underlying partitioned tensornetwork
function planargraph_partitionpair(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    return pair(partitionedge(ppg(bmpsc), parent(pe)))
end

#Sort (bottom to top) partitoonedges between pair of partitions in the planargraph
function planargraph_sorted_partitionedges(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pg = ppg(bmpsc)
    src_vs, dst_vs = vertices(pg, PartitionVertex(first(partitionpair))),
    vertices(pg, PartitionVertex(last(partitionpair)))
    es = reduce(
        vcat,
        [
            [src_v => dst_v for dst_v in intersect(neighbors(pg, src_v), dst_vs)] for
            src_v in src_vs
        ],
    )
    es = sort(NamedEdge.(es); by = x -> findfirst(isequal(src(x)), src_vs))
    return PartitionEdge.(es)
end

#Constructor, inserts missing edge in the planar graph to ensure each partition is connected 
#allowing the code to work for arbitrary grids and not just square grids
function BoundaryMPSCache(
    bpc::BeliefPropagationCache;
    grouping_function::Function = v -> first(v),
    group_sorting_function::Function = v -> last(v),
    message_rank::Int64 = default_boundarymps_message_rank(tensornetwork(bpc)),
)
    bpc = insert_pseudo_planar_edges(bpc; grouping_function)
    planar_graph = partitioned_graph(bpc)
    vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
    vertex_groups = map(x -> sort(x; by = group_sorting_function), vertex_groups)
    ppg = PartitionedGraph(planar_graph, vertex_groups)
    bmpsc = BoundaryMPSCache(bpc, ppg, message_rank)
    return set_interpartition_messages(bmpsc)
end

function BoundaryMPSCache(tn, args...; kwargs...)
    return BoundaryMPSCache(BeliefPropagationCache(tn, args...); kwargs...)
end

#Functions to get the parellel partitionedges sitting above and below a partitionedge
function partitionedges_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe))
    pe_pos = only(findall(x -> x == pe, pes))
    return PartitionEdge[pes[i] for i = (pe_pos+1):length(pes)]
end

function partitionedges_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe))
    pe_pos = only(findall(x -> x == pe, pes))
    return PartitionEdge[pes[i] for i = 1:(pe_pos-1)]
end

function partitionedge_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes_above = partitionedges_above(bmpsc, pe)
    isempty(pes_above) && return nothing
    return first(pes_above)
end

function partitionedge_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes_below = partitionedges_below(bmpsc, pe)
    isempty(pes_below) && return nothing
    return last(pes_below)
end

#Given a sequence of message tensor updates within a partition, get the sequence of gauge moves on the interpartition
# needed to move the MPS gauge from the start of the sequence to the end of the sequence
function mps_gauge_update_sequence(
    bmpsc::BoundaryMPSCache,
    partition_update_seq::Vector{<:PartitionEdge},
    partition_pair::Pair,
)
    vs = unique(reduce(vcat, [[src(pe), dst(pe)] for pe in parent.(partition_update_seq)]))
    g = planargraph(bmpsc)
    dst_vs = planargraph_vertices(bmpsc, last(partition_pair))
    pe_sequence = [v => intersect(neighbors(g, v), dst_vs) for v in vs]
    pe_sequence = filter(x -> !isempty(last(x)), pe_sequence)
    pe_sequence = map(x -> first(x) => only(last(x)), pe_sequence)
    return [
        (PartitionEdge(pe_sequence[i]), PartitionEdge(pe_sequence[i+1])) for
        i = 1:(length(pe_sequence)-1)
    ]
end

#Returns the sequence of pairs of partitionedges that need to be updated to move the MPS gauge between regions
function mps_gauge_update_sequence(
    bmpsc::BoundaryMPSCache,
    pe_region1::Vector{<:PartitionEdge},
    pe_region2::Vector{<:PartitionEdge},
)
    issetequal(pe_region1, pe_region2) && return []
    partitionpair = planargraph_partitionpair(bmpsc, first(pe_region2))
    seq = partition_update_sequence(
        bmpsc,
        parent.(src.(pe_region1)),
        parent.(src.(pe_region2)),
    )
    return mps_gauge_update_sequence(bmpsc, seq, partitionpair)
end

function mps_gauge_update_sequence(
    bmpsc::BoundaryMPSCache,
    pe_region::Vector{<:PartitionEdge},
)
    partitionpair = planargraph_partitionpair(bmpsc, first(pe_region))
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    return mps_gauge_update_sequence(bmpsc, pes, pe_region)
end

function mps_gauge_update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    return mps_gauge_update_sequence(bmpsc, [pe])
end

#Initialise all the interpartition message tensors
function set_interpartition_messages(
    bmpsc::BoundaryMPSCache,
    partitionpairs::Vector{<:Pair},
)
    bmpsc = copy(bmpsc)
    ms = messages(bmpsc)
    for partitionpair in partitionpairs
        pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
        for pe in pes
            set!(ms, pe, ITensor[dense(delta(linkinds(bmpsc, pe)))])
        end
        for i = 1:(length(pes)-1)
            virt_dim = virtual_index_dimension(bmpsc, pes[i], pes[i+1])
            ind = Index(virt_dim, "m$(i)$(i+1)")
            m1, m2 = only(ms[pes[i]]), only(ms[pes[i+1]])
            set!(ms, pes[i], ITensor[m1*delta(ind)])
            set!(ms, pes[i+1], ITensor[m2*delta(ind)])
        end
    end
    return bmpsc
end

function set_interpartition_messages(bmpsc::BoundaryMPSCache)
    partitionpairs = pair.(partitionedges(ppg(bmpsc)))
    return set_interpartition_messages(
        bmpsc,
        vcat(partitionpairs, reverse.(partitionpairs)),
    )
end

#Switch the message tensors on partition edges with their reverse (and dagger them)
function switch_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    bmpsc = copy(bmpsc)
    ms = messages(bmpsc)
    me, mer = message(bmpsc, pe), message(bmpsc, reverse(pe))
    set!(ms, pe, dag.(mer))
    set!(ms, reverse(pe), dag.(me))
    return bmpsc
end

function switch_messages(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    for pe in planargraph_sorted_partitionedges(bmpsc, partitionpair)
        bmpsc = switch_message(bmpsc, pe)
    end
    return bmpsc
end

#Get sequence necessary to update all message tensors in a partition
function partition_update_sequence(bmpsc::BoundaryMPSCache, partition)
    vs = planargraph_vertices(bmpsc, partition)
    return vcat(
        partition_update_sequence(bmpsc, [first(vs)]),
        partition_update_sequence(bmpsc, [last(vs)]),
    )
end

#Get sequence necessary to move correct message tensors in a partition from region1 to region2
function partition_update_sequence(
    bmpsc::BoundaryMPSCache,
    region1::Vector,
    region2::Vector,
)
    issetequal(region1, region2) && return PartitionEdge[]
    pv = planargraph_partition(bmpsc, first(region2))
    g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
    st = steiner_tree(g, union(region1, region2))
    path = post_order_dfs_edges(st, first(region2))
    path = filter(e -> !((src(e) ∈ region2) && (dst(e) ∈ region2)), path)
    return PartitionEdge.(path)
end

#Get sequence necessary to move correct message tensors to a region
function partition_update_sequence(bmpsc::BoundaryMPSCache, region::Vector)
    pv = planargraph_partition(bmpsc, first(region))
    return partition_update_sequence(bmpsc, planargraph_vertices(bmpsc, pv), region)
end

#Update all messages tensors within a partition by finding the path needed
function partition_update(bmpsc::BoundaryMPSCache, args...)
    return update(
        Algorithm("bp"),
        bmpsc,
        partition_update_sequence(bmpsc, args...);
        message_update_function_kwargs = (; normalize = false),
    )
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2 
function gauge_step(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    pe1::PartitionEdge,
    pe2::PartitionEdge;
    kwargs...,
)
    bmpsc = copy(bmpsc)
    ms = messages(bmpsc)
    m1, m2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
    @assert !isempty(commoninds(m1, m2))
    left_inds = uniqueinds(m1, m2)
    m1, Y = factorize(m1, left_inds; ortho = "left", kwargs...)
    m2 = m2 * Y
    set!(ms, pe1, ITensor[m1])
    set!(ms, pe2, ITensor[m2])
    return bmpsc
end

#Move the biorthogonality centre one step on an interpartition from the partition edge pe1 (and its reverse) to that on pe2 
function gauge_step(
    alg::Algorithm"biorthogonal",
    bmpsc::BoundaryMPSCache,
    pe1::PartitionEdge,
    pe2::PartitionEdge,
)
    bmpsc = copy(bmpsc)
    ms = messages(bmpsc)

    m1, m1r = only(message(bmpsc, pe1)), only(message(bmpsc, reverse(pe1)))
    m2, m2r = only(message(bmpsc, pe2)), only(message(bmpsc, reverse(pe2)))
    top_cind, bottom_cind = commonind(m1, m2), commonind(m1r, m2r)
    m1_siteinds, m2_siteinds = commoninds(m1, m1r), commoninds(m2, m2r)
    top_ncind = setdiff(inds(m1), [m1_siteinds; top_cind])
    bottom_ncind = setdiff(inds(m1r), [m1_siteinds; bottom_cind])

    E = if isempty(top_ncind)
        m1 * m1r
    else
        m1 * replaceind(m1r, only(bottom_ncind), only(top_ncind))
    end
    U, S, V = svd(E, bottom_cind; alg = "recursive")

    S_sqrtinv = map_diag(x -> pinv(sqrt(x)), S)
    S_sqrt = map_diag(x -> sqrt(x), S)

    m1 = replaceind((m1 * dag(V)) * S_sqrtinv, commonind(U, S), top_cind)
    m1r = replaceind((m1r * dag(U)) * S_sqrtinv, commonind(V, S), bottom_cind)
    m2 = replaceind((m2 * V) * S_sqrt, commonind(U, S), top_cind)
    m2r = replaceind((m2r * U) * S_sqrt, commonind(V, S), bottom_cind)
    set!(ms, pe1, ITensor[m1])
    set!(ms, reverse(pe1), ITensor[m1r])
    set!(ms, pe2, ITensor[m2])
    set!(ms, reverse(pe2), ITensor[m2r])

    return bmpsc
end

#Move the orthogonality / biorthogonality centre on an interpartition via a sequence of steps between message tensors
function ITensorNetworks.gauge_walk(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    seq::Vector;
    kwargs...,
)
    for (pe1, pe2) in seq
        bmpsc = gauge_step(alg::Algorithm, bmpsc, pe1, pe2; kwargs...)
    end
    return bmpsc
end

function gauge(alg::Algorithm, bmpsc::BoundaryMPSCache, args...; kwargs...)
    return gauge_walk(alg, bmpsc, mps_gauge_update_sequence(bmpsc, args...); kwargs...)
end

#Move the orthogonality centre on an interpartition to the message tensor on pe or between two pes
function ITensorNetworks.orthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
    return gauge(Algorithm("orthogonal"), bmpsc, args...; kwargs...)
end

#Move the biorthogonality centre on an interpartition to the message tensor or between two pes
function biorthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
    return gauge(Algorithm("biorthogonal"), bmpsc, args...; kwargs...)
end

default_inserter_transform(alg::Algorithm"biorthogonal") = identity
default_inserter_transform(alg::Algorithm"orthogonal") = dag
default_region_transform(alg::Algorithm"biorthogonal") = identity
default_region_transform(alg::Algorithm"orthogonal") = reverse

#Default inserter for the MPS fitting (one and two-site support)
function default_inserter(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    update_pe_region::Vector{<:PartitionEdge},
    ms::Vector{ITensor};
    inserter_transform = default_inserter_transform(alg),
    region_transform = default_region_transform(alg),
    nsites::Int64 = 1,
    cutoff = 1e-12,
    normalize = true,
)
    update_pe_region = region_transform.(update_pe_region)
    m = contract(ms; sequence = "automatic")
    if normalize
        m /= norm(m)
    end
    if nsites == 1
        bmpsc = set_message(bmpsc, only(update_pe_region), ITensor[inserter_transform(m)])
    elseif nsites == 2
        pe1, pe2 = first(update_pe_region), last(update_pe_region)
        me1, me2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
        upper_inds, cind = uniqueinds(me1, me2), commonind(me1, me2)
        me1, me2 = factorize(
            m,
            upper_inds;
            tags = tags(cind),
            cutoff,
            maxdim = maximum_virtual_dimension(bmpsc),
        )
        bmpsc = set_message(bmpsc, pe1, ITensor[inserter_transform(me1)])
        bmpsc = set_message(bmpsc, pe2, ITensor[inserter_transform(me2)])
    else
        error("Nsites > 2 not supported at the moment for Boundary MPS updating")
    end
    return bmpsc
end

#Default updater for the MPS fitting
function default_updater(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    prev_pe_region,
    update_pe_region,
)
    if !isnothing(prev_pe_region)
        bmpsc = gauge(alg, bmpsc, reverse.(prev_pe_region), reverse.(update_pe_region))
        bmpsc = partition_update(
            bmpsc,
            parent.(src.(prev_pe_region)),
            parent.(src.(update_pe_region)),
        )
    else
        bmpsc = gauge(alg, bmpsc, reverse.(update_pe_region))
        bmpsc = partition_update(bmpsc, parent.(src.(update_pe_region)))
    end
    return bmpsc
end

#Default extracter for the MPS fitting (1 and two-site support)
function default_extracter(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    update_pe_region::Vector{<:PartitionEdge};
    nsites::Int64 = 1,
)
    nsites == 1 && return updated_message(
        bmpsc,
        only(update_pe_region);
        message_update_function_kwargs = (; normalize = false),
    )
    if nsites == 2
        pv1, pv2 = src(first(update_pe_region)), src(last(update_pe_region))
        partition = planargraph_partition(bmpsc, parent(pv1))
        g = subgraph(planargraph(bmpsc), planargraph_vertices(bmpsc, partition))
        path = a_star(g, parent(pv1), parent(pv2))
        pvs = PartitionVertex.(vcat(src.(path), [parent(pv2)]))
        local_tensors = factors(bmpsc, pvs)
        ms = incoming_messages(bmpsc, pvs; ignore_edges = reverse.(update_pe_region))
        return ITensor[local_tensors; ms]
    else
        error("Nsites > 2 not supported at the moment")
    end
end

function ITensors.commonind(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
    m1, m2 = message(bmpsc, pe1), message(bmpsc, pe2)
    return commonind(only(m1), only(m2))
end

#Transformers for switching the virtual index of message tensors on boundary of pe_region
# to those of their reverse
function virtual_index_transformers(
    bmpsc::BoundaryMPSCache,
    pe_region::Vector{<:PartitionEdge},
)
    partitionpair = planargraph_partitionpair(bmpsc, first(pe_region))
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    sorted_pes = sort(pe_region; by = pe -> findfirst(x -> x == pe, pes))
    pe1, pe2 = first(sorted_pes), last(sorted_pes)
    pe_a, pe_b = partitionedge_above(bmpsc, pe2), partitionedge_below(bmpsc, pe1)
    transformers = ITensor[]
    if !isnothing(pe_b)
        transformers = [
            transformers
            delta(
                commonind(bmpsc, pe_b, pe1),
                commonind(bmpsc, reverse(pe_b), reverse(pe1)),
            )
        ]
    end
    if !isnothing(pe_a)
        transformers = [
            transformers
            delta(
                commonind(bmpsc, pe_a, pe2),
                commonind(bmpsc, reverse(pe_a), reverse(pe2)),
            )
        ]
    end
    return transformers
end

#Biorthogonal extracter for MPS fitting (needs virtual index transformer)
function default_extracter(
    alg::Algorithm"biorthogonal",
    bmpsc::BoundaryMPSCache,
    update_pe_region::Vector{<:PartitionEdge};
    nsites,
)
    ms = default_extracter(Algorithm("orthogonal"), bmpsc, update_pe_region; nsites)
    return [ms; virtual_index_transformers(bmpsc, update_pe_region)]
end

function default_cache_prep_function(
    alg::Algorithm"biorthogonal",
    bmpsc::BoundaryMPSCache,
    partitionpair,
)
    bmpsc = delete_partition_messages!(bmpsc, first(partitionpair))
    return bmpsc
end
function default_cache_prep_function(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    partitionpair,
)
    bmpsc = copy(bmpsc)
    bmpsc = delete_partition_messages!(bmpsc, first(partitionpair))
    return switch_messages(bmpsc, partitionpair)
end

#Cost functions
function default_costfunction(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    pe_region::Vector{<:PartitionEdge},
)
    bmpsc = copy(bmpsc)
    pe = first(pe_region)
    bmpsc = gauge(alg, bmpsc, reverse.(pe_region), [reverse(pe)])
    bmpsc = partition_update(bmpsc, parent.(src.(pe_region)), [parent(src(pe))])
    return region_scalar(bp_cache(bmpsc), src(pe)) / norm(only(message(bmpsc, pe)))
end

function default_costfunction(
    alg::Algorithm"biorthogonal",
    bmpsc::BoundaryMPSCache,
    pe_region::Vector{<:PartitionEdge};
    nsites::Int64 = 1,
)
    bmpsc = copy(bmpsc)
    pe = first(pe_region)
    bmpsc = gauge(alg, bmpsc, reverse.(pe_region), [reverse(pe)])
    bmpsc = partition_update(bmpsc, parent.(src.(pe_region)), [parent(src(pe))])
    ms = [only(message(bmpsc, pe)), only(message(bmpsc, reverse(pe)))]
    ms = [ms; virtual_index_transformers(bmpsc, [pe])]
    return region_scalar(bp_cache(bmpsc), src(pe)) / contract(ms; sequence = "automatic")[]
end

#Sequences
function update_sequence(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    partitionpair::Pair;
    nsites::Int64 = 1,
)
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    if nsites == 1
        return vcat([[pe] for pe in pes], [[pe] for pe in reverse(pes[2:(length(pes)-1)])])
    elseif nsites == 2
        seq = [[pes[i], pes[i+1]] for i = 1:(length(pes)-1)]
        #TODO: Why does this not work reversing the elements of seq?
        return seq
    end
end

function merge_internal_tensors(O::Union{MPS, MPO})
    internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])

    while !isempty(internal_inds)
        site = first(internal_inds)
        tensors = [O[i] for i in setdiff([i for i in 1:length(O)], [site])]
        if site != length(O)
            tensors[site] = tensors[site] * O[site]
        else
            tensors[site - 1] = tensors[site - 1] * O[site]
        end

        O = typeof(O)(tensors)

        internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])
    end
    return O
end

function ITensorMPS.MPO(bmpsc::BoundaryMPSCache, partition)
    sorted_vs = sort(planargraph_vertices(bmpsc, partition))
    ts = [only(factors(bmpsc, [v])) for v in sorted_vs]
    O = ITensorMPS.MPO(ts)
    #O = merge_internal_tensors(O)
    return O
end

function ITensorMPS.MPS(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    sorted_pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    ms = [only(message(bmpsc, pe)) for pe in sorted_pes]
    return ITensorMPS.MPS(ms)
end

function ITensorNetworks.truncate(bmpsc::BoundaryMPSCache, partitionpair::Pair; truncate_kwargs...)
    bmpsc = copy(bmpsc)
    M = ITensorMPS.MPS(bmpsc, partitionpair)
    M = ITensorMPS.truncate(M; truncate_kwargs...)
    return set_interpartition_message(bmpsc, M, partitionpair)
end

function set_interpartition_message(bmpsc::BoundaryMPSCache, M::Union{MPS, MPO}, partitionpair::Pair)
    bmpsc = copy(bmpsc)
    sorted_pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    ms = messages(bmpsc)
    for i in 1:length(M)
        set!(ms, sorted_pes[i], ITensor[M[i]])
    end
    return bmpsc
end


#Update all the message tensors on an interpartition via an n-site fitting procedure 
function ITensorNetworks.update(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    partitionpair::Pair;
    inserter = default_inserter,
    costfunction = default_costfunction,
    updater = default_updater,
    extracter = default_extracter,
    cache_prep_function = default_cache_prep_function,
    niters::Int64,
    tolerance,
    normalize = true,
    truncate_at_end = false,
    truncate_kwargs = (;),
    nsites::Int64 = 1,
)
    bmpsc = cache_prep_function(alg, bmpsc, partitionpair)
    update_seq = update_sequence(alg, bmpsc, partitionpair; nsites)
    prev_cf = 0
    for i = 1:niters
        cf = 0
        for (j, update_pe_region) in enumerate(update_seq)
            prev_pe_region = j == 1 ? nothing : update_seq[j-1]
            bmpsc = updater(alg, bmpsc, prev_pe_region, update_pe_region)
            ms = extracter(alg, bmpsc, update_pe_region; nsites)
            bmpsc = inserter(alg, bmpsc, update_pe_region, ms; nsites, normalize)
            cf += !isnothing(tolerance) ? costfunction(alg, bmpsc, update_pe_region) : 0.0
        end
        epsilon = abs(cf - prev_cf) / length(update_seq)
        if !isnothing(tolerance) && epsilon < tolerance
            bmpsc = cache_prep_function(alg, bmpsc, partitionpair)
            if truncate_at_end
                bmpsc = truncate(bmpsc, partitionpair; truncate_kwargs...)
            end
            return bmpsc
        else
            prev_cf = cf
        end
    end
    bmpsc = cache_prep_function(alg, bmpsc, partitionpair)
    if truncate_at_end
        bmpsc = truncate(bmpsc, partitionpair; truncate_kwargs...)
    end
    return bmpsc
end

function prev_partitionpair(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pppg = partitioned_graph(ppg(bmpsc))
    vns = neighbors(pppg, first(partitionpair))
    length(vns) == 1 && return nothing

    @assert length(vns) == 2
    v1, v2 = first(vns), last(vns)
    last(partitionpair) == v1 && return v2 => first(partitionpair)
    last(partitionpair) == v2 && return v1 => first(partitionpair)
end

function generic_apply(O::MPO, M::MPS; kwargs...)
    length(O) == length(M) && return ITensorMPS.apply(O, M; kwargs...)

    O_tensors = ITensor[]
    for i in 1:length(O)
        m_ind = filter(j -> !isempty(ITensors.commoninds(O[i], M[j])), [j for j in 1:length(M)])
        if isempty(m_ind)
            push!(O_tensors, O[i])
        else
            m_ind = only(m_ind)
            push!(O_tensors, O[i] * M[m_ind])
        end
    end
    O = ITensorNetwork([i for i in 1:length(O_tensors)], O_tensors)
    O = ITensorNetworks.combine_linkinds(O)
    O = ITensorMPS.MPS([O[v] for v in vertices(O)])
    O = merge_internal_tensors(O)
    return truncate(O; kwargs...)
end

#Update all the message tensors on an interpartition via the ITensorMPS apply function
function ITensorNetworks.update(
    alg::Algorithm"ITensorMPS",
    bmpsc::BoundaryMPSCache,
    partitionpair::Pair;
    cutoff::Number, 
    maxdim::Int,
    kwargs...
)
    prev_pp = prev_partitionpair(bmpsc, partitionpair)
    O = ITensorMPS.MPO(bmpsc, first(partitionpair))
    O = ITensorMPS.truncate(O; cutoff, maxdim)
    isnothing(prev_pp) && return set_interpartition_message(bmpsc, merge_internal_tensors(O), partitionpair)
    
    M = ITensorMPS.MPS(bmpsc, prev_pp)
    M_out = generic_apply(O, M; cutoff, maxdim)
    return set_interpartition_message(bmpsc, M_out, partitionpair)
end

#Environment support, assume all vertices live in the same partition for now
function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
    vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
    bmpsc = partition_update(bmpsc, vs)
    return environment(bp_cache(bmpsc), verts; kwargs...)
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, partition)
    partition_vs = planargraph_vertices(bmpsc, partition)
    bmpsc = partition_update(bmpsc, [first(partition_vs)], [last(partition_vs)])
    return region_scalar(bp_cache(bmpsc), PartitionVertex(last(partition_vs)))
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, verts::Vector)
    partitions = planargraph_partitions(bmpsc, parent.(partitionvertices(bmpsc, verts)))
    if length(partitions) == 1
        return region_scalar(bmpsc, only(partitions))
    end
    error("Contractions involving more than 1 partition not currently supported")
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    out = ITensor(one(Bool))
    for pe in pes
        out = (out * (only(message(bmpsc, pe)))) * only(message(bmpsc, reverse(pe)))
    end
    return out[]
end

function add_partitionedges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
    g = partitioned_graph(pg)
    g = add_edges(g, parent.(pes))
    return PartitionedGraph(
        unpartitioned_graph(pg),
        g,
        partitioned_vertices(pg),
        which_partition(pg),
    )
end

function add_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    pg = add_partitionedges(partitioned_tensornetwork(bpc), pes)
    return BeliefPropagationCache(pg, messages(bpc))
end

#Add partition edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function insert_pseudo_planar_edges(
    bpc::BeliefPropagationCache;
    grouping_function = v -> first(v),
)
    pg = partitioned_graph(bpc)
    partitions = unique(grouping_function.(collect(vertices(pg))))
    pseudo_edges = PartitionEdge[]
    for p in partitions
        vs = sort(filter(v -> grouping_function(v) == p, collect(vertices(pg))))
        for i = 1:(length(vs)-1)
            if vs[i] ∉ neighbors(pg, vs[i+1])
                push!(pseudo_edges, PartitionEdge(NamedEdge(vs[i] => vs[i+1])))
            end
        end
    end
    return add_partitionedges(bpc, pseudo_edges)
end

pair(pe::PartitionEdge) = parent(src(pe)) => parent(dst(pe))

function delete_partition_messages!(bmpsc::BoundaryMPSCache, partition)
    vs = sort(planargraph_vertices(bmpsc, partition))
    pes = partition_update_sequence(bmpsc, [first(vs)])
    pes = vcat(pes, reverse.(pes))
    return delete_messages!(bmpsc, pes)
end

function delete_partitionpair_messages!(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    return delete_messages!(bmpsc, pes)
end