# Note: original author of this file is Terry Stewart

import nengo
import numpy as np

from nengo.utils.filter_design import cont2discrete

class Unconvertible(Exception):
    pass

class ProbeNode(nengo.Node):
    def __init__(self, size_in):
        super(ProbeNode, self).__init__(
            output=self.collect,
            label='Probe(%d)' % size_in,
            size_in=size_in,
            size_out=size_in)
        self.data = []

    def collect(self, t, x):
        self.data.append(x)
        return x


def find_passthrough_nodes(model):
    nodes = [n for n in model.nodes if n.output is None]
    in_nodes = []
    out_nodes = []
    if isinstance(model, nengo.networks.EnsembleArray):
        in_nodes.append(model.input)
        out_nodes = [n for n in nodes if n not in in_nodes]
        nodes = []
    for net in model.networks:
        n, i, o = find_passthrough_nodes(net)
        nodes.extend(n)
        in_nodes.extend(i)
        out_nodes.extend(o)
    return nodes, in_nodes, out_nodes


def create_replacement(c_in, c_out, dt):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post_obj is c_out.pre_obj
    assert c_in.post_obj.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        def extract_filter_coeff(flt):
            if isinstance(flt, nengo.synapses.LinearFilter):
                num, den = flt.num, flt.den
                if flt.analog:
                    num, den, _ = cont2discrete((num, den), dt, method='gbt', alpha=1.0)
                    num = num.flatten()
                return (num, den)
            return None

        c_in_coeff = extract_filter_coeff(c_in.synapse)
        c_out_coeff = extract_filter_coeff(c_out.synapse)
        if (c_in_coeff is None) or (c_out_coeff is None):
            raise Unconvertible("Cannot merge two filters")

        delay_coeff = ([1], [1])

#        print(c_in_coeff, c_out_coeff, delay_coeff)

#        print(np.polymul(delay_coeff[0], c_out_coeff[0]))
#        print(np.polymul(delay_coeff[1], c_out_coeff[1]))

        synapse = nengo.synapses.LinearFilter(
            np.polymul(c_in_coeff[0], np.polymul(delay_coeff[0], c_out_coeff[0])),
            np.polymul(c_in_coeff[1], np.polymul(delay_coeff[1], c_out_coeff[1])), analog=False)

    function = c_in.function
    if c_out.function is not None:
        raise Unconvertible("Cannot remove a connection with a function")

    # compute the combined transform
    transform = np.dot(
        nengo.utils.builder.full_transform(c_out),
        nengo.utils.builder.full_transform(c_in))

    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.allclose(transform, 0):
        return None

    return nengo.Connection(
        c_in.pre_obj,
        c_out.post_obj,
        synapse=synapse,
        transform=transform,
        function=function,
        add_to_container=False)


def remove_nodes(objs, passthrough, original_conns, dt):
    inputs = {}
    outputs = {}
    for obj in objs:
        inputs[obj] = []
        outputs[obj] = []
        if isinstance(obj, nengo.Ensemble):
            inputs[obj.neurons] = []
            outputs[obj.neurons] = []
    for c in original_conns:
        inputs[c.post_obj].append(c)
        outputs[c.pre_obj].append(c)

    for n in passthrough:
        for c_in in inputs[n]:
            for c_out in outputs[n]:
                c = create_replacement(c_in, c_out, dt)
                if c is not None:
                    outputs[c_in.pre_obj].append(c)
                    inputs[c_out.post_obj].append(c)
        for c_in in inputs[n]:
            outputs[c_in.pre_obj].remove(c_in)
        for c_out in outputs[n]:
            inputs[c_out.post_obj].remove(c_out)
        del inputs[n]
        del outputs[n]

    conns = []
    for cs in inputs.values():
        conns.extend(cs)
    return conns


def preprocess(model, scale_transform_by_radius=False, dt=1e-3):
    network = nengo.Network(add_to_container=False)

    network.ensembles.extend(model.all_ensembles)
    network.nodes.extend(model.all_nodes)

    probes = {}
    probe_conns = []
    for p in model.all_probes:
        with network:
            node = ProbeNode(p.size_in)
            c = nengo.Connection(
                p.target, node, synapse=p.synapse, add_to_container=False)
            probes[p] = node
            probe_conns.append(c)

    conns = model.all_connections + probe_conns

    passthrough, input_nodes, output_nodes = find_passthrough_nodes(model)
    conns = remove_nodes(network.ensembles + network.nodes,
                         passthrough + input_nodes + output_nodes, conns, dt)
    for n in (passthrough + input_nodes + output_nodes):
        network.nodes.remove(n)

    network.connections.extend(conns)

    return network, probes

