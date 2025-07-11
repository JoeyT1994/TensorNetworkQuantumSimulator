using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using Statistics

function main()
    #Define the lattice
    g = TN.heavy_hexagonal_lattice(5,5)

    #Define the physical indices on each site
    s = ITN.siteinds("S=1/2", g)

    #Define an edge coloring
    ec = edge_color(g, 3)

    #Define the gate parametrs
    J = pi / 4
    θh = 0.4

    #Now the circuit
    Rx_layer = [("Rx", [v], θh) for v in collect(vertices(g))]
    Rzz_layer = []
    for edge_group in ec
        append!(Rzz_layer, ("Rzz", pair, 2*J) for pair in edge_group)
    end
    layer = vcat(Rx_layer, Rzz_layer)

    #Depth of the circuit and apply parameters
    no_trotter_steps = 20
    χ = 8
    apply_kwargs = (; cutoff = 1e-12, maxdim = χ)

    #Initial state
    ψt = ITensorNetwork(v -> "↑", s)

    #BP cache for norm of the network
    ψψ = build_bp_cache(ψt)

    #Do the evolution
    fidelities = []
    for i in 1:no_trotter_steps
        println("Applying gates for Trotter Step $(i)")
        ψt, ψψ, errs = apply(layer, ψt, ψψ; apply_kwargs)
        fidelity = prod(1.0 .- errs)
        println("Layer fidelity was $(fidelity)")
        push!(fidelities, fidelity)
    end

    println("Total final fidelity is $(prod(fidelities))")
    ntwo_site_gates = length(edges(g)) * no_trotter_steps
    println("Avg gate fidelity is $((prod(fidelities)) ^((1 / ntwo_site_gates)))")

    central_site = (11,5)

    #Use BP to get an observable
    sz_bp = expect(ψt, [("Z", [central_site])]; alg = "bp")
    println("BP measured magnetisation on central site is $(only(sz_bp))")

    #Use boundary MPS to get same observab;e
    message_rank = 10
    sz_bmps = expect(ψt, [("Z", [central_site])]; alg = "boundarymps", message_rank)

    println("Boundary MPS measured magnetisation on central site with rank $(message_rank) MPSs is $(only(sz_bp))")

    #Sample from q(x) and get p(x) / q(x) for each sample too
    nsamples = 100
    bitstrings = TN.sample_directly_certified(ψt, nsamples; norm_message_rank = 8)

    st_dev = Statistics.std(first.(bitstrings))
    println("Standard deviation of p(x) / q(x) is $(st_dev)")

    #Measure observable with sample approach
    sampled_sz = sum([first(b) * (-2*last(b)[central_site] + 1) for b in bitstrings]) / Statistics.sum(first.(bitstrings))
    println("Importance sampled value for magnetisation is $(sampled_sz)")

end

main()