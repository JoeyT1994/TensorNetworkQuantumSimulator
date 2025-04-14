# conversion of a tuple circuit to an ITensor circuit
function toitensor(circuit, sinds::IndsNetwork)
    return [toitensor(gate, sinds) for gate in circuit]
end

function _ispaulistring(string::String)
    return all(s ∈ ['X', 'Y', 'X', 'x', 'y', 'x'] for s in string)
end

function _takes_theta_argument(string::String)
    return string ∈ ["Rx", "Ry", "Rz", "Cx", "Cy", "Rz"]
end


function _takes_phi_argument(string::String)
    return string ∈ ["Rxx", "Ryy", "Rzz"]
end

# conversion of the tuple gate to an ITensor
function toitensor(gate::Tuple, sinds::IndsNetwork)

    gate_symbol = gate[1]
    gate_inds = gate[2]
    # if it is a NamedEdge, we need to convert it to a tuple
    gate_inds = _ensuretuple(gate_inds)
    s_inds = [only(sinds[v]) for v in gate_inds]

    all(map(sind -> dim(sind) == 4, s_inds)) &&
        return toitensor_heisenberg(gate_symbol, gate[3], sinds)

    if _ispaulistring(gate_symbol)
        gate =
            prod(ITensors.op(string(op), sind) for (op, sind) in zip(gate_symbol, s_inds))
    elseif length(gate) == 2
        gate = ITensors.op(gate_symbol, s_inds...)
    elseif _takes_theta_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; θ = gate[3])
    elseif _takes_phi_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; ϕ = 0.5 * gate[3])
    else
        throw(ArgumentError("Wrong gate format"))
    end

    return gate

end

function toitensor_heisenberg(generator, θ, indices)
    @assert first(generator) == 'R'
    generator = generator[2:length(generator)]
    @assert _ispaulistring(generator)
    U = paulirotationmatrix(generator, θ)
    U = PP.calculateptm(U, heisenberg = true)

    # check for physical dimension matching
    # TODO

    # define legs of the tensor
    legs = (indices..., [ind' for ind in indices]...)

    # create the ITensor
    return itensor(transpose(U), legs)
end

# conversion retruns the gate itself if it is already
function toitensor(gate::ITensor, sinds::IndsNetwork)
    return gate
end

# conversion of the gate indices to a tuple
function _ensuretuple(gate_inds::Union{Tuple,AbstractArray})
    return gate_inds
end

function _ensuretuple(gate_inds::NamedEdge)
    return (gate_inds.src, gate_inds.dst)
end
