using LinearAlgebra
using StatsBase

using Dictionaries: Dictionary, set!

using Graphs: simplecycles_limited_length, has_edge, SimpleGraph, center, steiner_tree, is_tree

using SimpleGraphConverter
using SimpleGraphAlgorithms: edge_color

using NamedGraphs
using NamedGraphs:
    AbstractNamedGraph,
    AbstractGraph,
    AbstractEdge,
    position_graph,
    rename_vertices,
    edges,
    vertextype,
    add_vertex!,
    neighbors,
    edgeinduced_subgraphs_no_leaves,
    unique_cyclesubgraphs_limited_length
using NamedGraphs.GraphsExtensions:
    src,
    dst,
    subgraph,
    is_connected,
    degree,
    add_edge,
    a_star,
    add_edge!,
    edgetype,
    leaf_vertices,
    post_order_dfs_edges,
    decorate_graph_edges,
    add_vertex!,
    add_vertex,
    rem_edge,
    rem_vertex,
    add_edges,
    rem_vertices

using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using TensorOperations

using ITensors: ITensors
using ITensors: Index, ITensor, hasqns, combinedind, combiner, replaceinds, sim, onehot, delta, plev, dense, unioninds, uniqueinds, commonind, commoninds, replaceind, datatype, inds, dag, noprime, factorize_svd, prime, hascommoninds, inner, itensor, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, denseblocks
using ITensorMPS

using ITensorNetworks: ITensorNetworks
using ITensorNetworks:
    AbstractITensorNetwork,
    AbstractIndsNetwork,
    Indices,
    IndsNetwork,
    ITensorNetwork,
    Algorithm,
    vertices,
    dim,
    neighbor_vertices,
    contraction_sequence,
    group,
    linkinds,
    generic_state,
    setindex_preserve_graph!,
    edge_tag,
    tree_orthogonalize,
    gauge_walk,
    maxlinkdim,
    default_cache_construction_kwargs

using Adapt: adapt

using ITensorNetworks.ITensorsExtensions: map_eigvals

using EinExprs: Greedy

import PauliPropagation
const PP = PauliPropagation

using TypeParameterAccessors: unspecify_type_parameters
