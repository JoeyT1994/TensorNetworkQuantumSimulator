# Boundary MPS helpers for checking graph formats
function is_line_graph(g::AbstractGraph)
    vs = collect(vertices(g))
    nvs = length(vs)
    length(vs) == 1 && return true
    !is_tree(g) && return false
    ds = sort([degree(g, v) for v in vs])
    ds != vcat([1, 1], [2 for d in 1:(nvs - 2)]) && return false
    return true
end

function is_ring_graph(g::AbstractGraph)
    isempty(edges(g)) && return false
    g_mod = rem_edge(g, first(edges(g)))
    return is_line_graph(g_mod)
end

function pseudo_sqrt_inv_sqrt(M::ITensor; cutoff = 10 * eps(real(scalartype(M))))
    @assert length(inds(M)) == 2
    Q, D, Qdag = ITensorNetworks.ITensorsExtensions.eigendecomp(M, inds(M)[1], inds(M)[2]; ishermitian = true)
    D_sqrt = ITensorNetworks.ITensorsExtensions.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : sqrt(x), D)
    D_inv_sqrt = ITensorNetworks.ITensorsExtensions.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : inv(sqrt(x)), D)
    M_sqrt = Q * D_sqrt * Qdag
    M_inv_sqrt = Q * D_inv_sqrt * Qdag
    return M_sqrt, M_inv_sqrt
end

#Function for checking the correct algorithm is being used for the given cache type and functionality
function algorithm_check(tns::Union{AbstractBeliefPropagationCache, TensorNetworkState}, f::String, alg)
    if alg == "bp"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'bp' algorithm, got $(typeof(tns))")
        end

        if f ∈ ["sample"]
            error("BP-based contraction not supported for this functionality yet")
        end
    elseif alg == "loopcorrections"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'loop correctiom' algorithm, got $(typeof(tns))")
        end

        if f ∈ ["normalize", "expect", "entanglement", "sample", "truncate"]
            return error("Loop correction-based contraction not supported for this functionality yet")
        end
    elseif alg == "boundarymps"
        if !((tns isa BoundaryMPSCache) || (tns isa TensorNetworkState))
            return error("Expected BoundaryMPSCache or TensorNetworkState for 'boundarymps' algorithm, got $(typeof(tns))")
        end
        if f ∈ ["normalize", "entanglement"]
            return error("boundarymps contraction not supported for this functionality yet")
        end
    elseif alg == "exact"
        if f ∈ ["normalize", "entanglement", "sample", "truncate"]
            return error("exact contraction not supported for this functionality yet")
        end
    elseif alg ∉ ["exact", "bp", "loopcorrections", "boundarymps"]
        return error("Unrecognized algorithm specified. Must be one of 'exact', 'bp', 'loopcorrections', or 'boundarymps'")
    else
        return nothing
    end
end

default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(bmps_cache::BoundaryMPSCache) = "boundarymps"
default_alg(any) = error("You must specify a contraction algorithm. Currently supported: exact, bp and boundarymps.")
