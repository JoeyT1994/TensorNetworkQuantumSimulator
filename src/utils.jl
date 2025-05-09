getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)

function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg = "bp")
    return val
end

## Truncate a tensor network down to a maximum bond dimension
"""
    truncate(ψ::ITensorNetwork; maxdim, cutoff=nothing, bp_update_kwargs= (...))

Truncate the ITensorNetwork `ψ` to a maximum bond dimension `maxdim` using the specified singular value cutoff.
"""
function ITensorNetworks.truncate(
    ψ::ITensorNetwork;
    kwargs...,
)
    ψ_vidal = VidalITensorNetwork(ψ; kwargs...)
    return ITensorNetwork(ψ_vidal)
end
# 
