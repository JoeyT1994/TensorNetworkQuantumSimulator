using ITensors: random_itensor

struct TensorNetworkState{V} <: AbstractITensorNetwork{V}
    tensornetwork::ITensorNetwork{V}
    siteinds::Dictionary
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
siteinds(tns::TensorNetworkState) = tns.siteinds

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)), copy(siteinds(tns)))

siteinds(tn::ITensorNetwork) = Dictionary(collect(vertices(tn)), [uniqueinds(tn, v) for v in collect(vertices(tn))])
TensorNetworkState(tn::ITensorNetwork) = TensorNetworkState(tn, siteinds(tn))

#Forward onto the itn
for f in [
    :(ITensorNetworks.underlying_graph),
    :(ITensorNetworks.data_graph_type),
    :(ITensorNetworks.data_graph),
    :(ITensors.datatype),
    :(ITensors.NDTensors.scalartype),
]
@eval begin
    function $f(tns::TensorNetworkState, args...; kwargs...)
        return $f(tensornetwork(tns), args...; kwargs...)
    end
end
end

#Forward onto the underlying_graph
for f in [
    :(NamedGraphs.edgeinduced_subgraphs_no_leaves)
]
@eval begin
    function $f(tns::TensorNetworkState, args...; kwargs...)
        return $f(ITensorNetworks.underlying_graph(tensornetwork(tns)), args...; kwargs...)
    end
end
end

siteinds(tns::TensorNetworkState, v) = siteinds(tns)[v]

function ITensorNetworks.data_graph_type(TNS::Type{<:TensorNetworkState})
    return ITensorNetworks.data_graph_type(fieldtype(TNS, :tensornetwork))
end

function ITensorNetworks.uniqueinds(tns::TensorNetworkState, v)
    is = ITensorNetworks.uniqueinds(tensornetwork(tns), v)
    is isa Vector{<:Index} && return is
    return Index[i for i in is]
end

ITensorNetworks.uniqueinds(tns::TensorNetworkState, edge::AbstractEdge) = ITensorNetworks.uniqueinds(tensornetwork(tns), edge)


function Base.setindex!(tns::TensorNetworkState, value, v)
    setindex!(tensornetwork(tns), value, v)
    sinds = siteinds(tns)
    for vn in vcat(neighbors(tns, v), [v])
        set!(sinds, vn, uniqueinds(tns, vn))
    end
    return tns
end

function ITensorNetworks.setindex_preserve_graph!(tns::TensorNetworkState, value, v)
    ITensorNetworks.setindex_preserve_graph!(tensornetwork(tns), value, v)
    sinds = siteinds(tns)
    for vn in vcat(neighbors(tns, v), [v])
        set!(sinds, vn, uniqueinds(tns, vn))
    end
    return tns
end

function setindex_preserve_all!(tns::TensorNetworkState, value, v)
    ITensorNetworks.setindex_preserve_graph!(tensornetwork(tns), value, v)
    return tns
end

setindex_preserve_all!(tn::ITensorNetwork, value, v) = ITensorNetworks.setindex_preserve_graph!(tn, value, v)

function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    factors = ITensor[]
    for v in verts
        sinds = siteinds(tns, v)
        tnv = tns[v]
        tnv_dag = dag(prime(tnv))
        if op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds) 
            append!(factors, ITensor[tnv, tnv_dag])
        else
            op = adapt(datatype(tnv))(ITensors.op(op_strings(v), only(sinds)))
            append!(factors, ITensor[tnv, tnv_dag, op])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v) = norm_factors(tns, [v])
bp_factors(tns::TensorNetworkState, v) = norm_factors(tns, v)

function default_message(tns::TensorNetworkState, edge::AbstractEdge)
    linds = linkinds(tns, edge)
    return adapt(datatype(tns))(denseblocks(delta(vcat(linds, prime(dag(linds))))))
end

function random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype); bond_dimension::Int = 1)
    vs = collect(vertices(g))
    siteinds = Dictionary(vs, [Index[Index(d, sitetype)] for v in vs])
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = ITensorNetwork(g)
    for v in vs
       is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g,v)])
       tn[v] = random_itensor(eltype, is)
    end
    return TensorNetworkState(tn, siteinds)
end

function tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype))
    vs = collect(vertices(g))
    siteinds = Dictionary(vs, [Index[Index(d, sitetype)] for v in vs])
    tn = ITensorNetwork(g)
    for v in vs
        tn[v] = adapt(eltype)(ITensors.state(f(v), only(siteinds[v])))
    end

    l = Dict(e => Index(1) for e in edges(g))
    for e in edges(g)
        tn[src(e)] *= onehot(eltype, l[e] => 1)
        tn[dst(e)] *= onehot(eltype, l[e] => 1)
    end
    return TensorNetworkState(tn, siteinds)
end

function random_tensornetworkstate(g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype); bond_dimension::Int = 1)
    return random_tensornetworkstate(Float64, sitetype, d; bond_dimension)
end

function tensornetworkstate(f::Function, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype))
    return tensornetworkstate(Float64, f, sitetype, d)
end

function site_dimension(sitetype::String)
    sitetype ∈ ["S=1/2", "Qubit", "Spin 1/2", "Spin Half", "Spin half"] && return 2
    sitetype ∈ ["Qutrit", "S=1", "Spin 1"] && return 3
    error("Don't know what physical space that site type should be")
end

    