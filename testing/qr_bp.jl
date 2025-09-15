using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: AbstractBeliefPropagationCache, BeliefPropagationCache

using ITensors: Algorithm, @Algorithm_str

using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges

using Random

using CUDA

function main()
    g = named_grid((5,5))
    s = siteinds("S=1/2", g)

    Random.seed!(1234)
    ψ0 = ITensorNetworks.random_tensornetwork(ComplexF32, s; link_space = 2)
    #ψ0 = ITensorNetworks.ITensorNetwork(v -> "X+", s)

    # ψ_bpc = ITensorNetworks.BeliefPropagationCache(ψ)
    # initialize_sqrt_bp_cache!(ψ_bpc)

    # ψ_bpc = ITensorNetworks.update(ψ_bpc; message_update_alg = Algorithm("qr"), maxiter = 25, tol = 1e-10, verbose=  true)

    # ψψ_bpc = bpc_from_sqrt_bpc(ψ_bpc)

    # @show expect(ψψ_bpc, ("Z", [(1,1)]))

    # @show expect(ψ, ("Z", [(1,1)]); alg = "bp")

    h, J = -1.0, -1.0
    no_trotter_steps = 10
    δt = 0.1

    #Do a 7-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 4)
    append!(layer, ("Rz", [v], h*δt) for v in vertices(g))
    for colored_edges in ec
        append!(layer, ("Rxx", pair, 2*J*δt) for pair in colored_edges)
    end
    append!(layer, ("Rz", [v], h*δt) for v in vertices(g))

    ψ = CUDA.cu(copy(ψ0))

    t1 = time()
    for i in 1:no_trotter_steps
        ψ, errs = TensorNetworkQuantumSimulator.apply(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1e-10))

        #println("    Max bond dimension: $(maxlinkdim(ψ))")
        #println("    Maximum Gate error for layer was $(maximum(errs))")
    end
    t2 = time()

    println("Time evo took $(t2-t1) secs")

    @show expect(ψ, ("Z", [(1,1)]); alg = "bp")

    ψ = CUDA.cu(copy(ψ0))
    t1 = time()
    for i in 1:no_trotter_steps
        ψ, errs = TensorNetworkQuantumSimulator.apply_via_sqrt(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1e-10))

        #println("    Max bond dimension: $(maxlinkdim(ψ))")
        #println("    Maximum Gate error for layer was $(maximum(errs))")
    end
    t2 = time()

    println("Time evo took $(t2-t1) secs")

    @show expect(ψ, ("Z", [(1,1)]); alg = "bp")
end

main()