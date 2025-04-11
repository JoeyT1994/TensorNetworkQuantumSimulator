getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)


getphysicaldim(indices::IndsNetwork) = dim(first(indices.data_graph.vertex_data.values))
getphysicaldim(indices::AbstractVector{<:Index}) = first(indices)
getphysicaldim(index::Index) = dim(index)


## 
function getmode(indices::IndsNetwork)
    d = getphysicaldim(indices)
    if d == 2
        return "Schrödinger"
    elseif d == 4
        return "Heisenberg"
    else
        throwdimensionerror()
    end
end

function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg="bp")
    return val
end


function topologytograph(topology)
    # TODO: adapt this to named graphs with non-integer labels
    # find number of vertices
    nq = maximum(maximum.(topology))
    adjm = zeros(Int, nq, nq)
    for (ii, jj) in topology
        adjm[ii, jj] = adjm[jj, ii] = 1
    end
    return NamedGraph(SimpleGraph(adjm))
end


function graphtotopology(g)
    return [[edge.src, edge.dst] for edge in edges(g)]
end

stringtosymbols(str) = [Symbol(s) for s in str]

function throwdimensionerror()
    throw(ArgumentError("Only physical dimensions 2 and 4 are supported."))
end

function get_global_cache_update_kwargs(alg::Algorithm)
    alg == Algorithm("bp") && return get_global_bp_update_kwargs()
    alg == Algorithm("boundarymps") && return get_global_boundarymps_update_kwargs()
    error("No update parameters known for that algorithm")
end

function ITensors.scalar(
    alg::Algorithm"loopcorrections",
    tn::AbstractITensorNetwork;
    max_configuration_size::Int64,
    (cache!)=nothing,
    cache_construction_kwargs=default_cache_construction_kwargs(Algorithm("bp"), tn),
    update_cache=isnothing(cache!),
    cache_update_kwargs=default_cache_update_kwargs(Algorithm("bp")),
  )
    if isnothing(cache!)
      cache! = Ref(cache(Algorithm("bp"), tn; cache_construction_kwargs...))
    end
  
    if update_cache
      cache![] = update(cache![]; cache_update_kwargs...)
    end
  
    return scalar(cache![]; alg, max_configuration_size)
  end
# 