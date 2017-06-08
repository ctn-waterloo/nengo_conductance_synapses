#!/usr/bin/env python3

#    Conductance based synapses in Nengo
#    Copyright (C) 2017  Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import nengo
import numpy as np

from . import lif_cond
from . import lif_cond_utils


def sim_if_cond_exp(decoders,
                    activities,
                    encoders,
                    direct,
                    scale_E,
                    scale_I,
                    model,
                    dt=1e-4,
                    use_jbias=False,
                    use_factorised_weights=False):
    """
    Function which builds the simulation functor that can be pluged into a
    Nengo Node object.

    decoders: list of decoders of the pre-population for the chosen function for
    each connection.
    activity: matrix containing the pre-synaptic neuronactivity for
    each of the evaluation points. This matrix is used to decode for the
    biases. Since the biases are a constant function, the evaluation
    points themselves are not required.
    encoders: list of encoders to be used for each connection.
    direct: list of flags indicating whether this particular connection should
    directly target the current.
    bias: bias values that should be used for this population.
    gain: gain values that should be used for this population.
    model: instance of the LifCond class defined above. If None
    is given, a new instance with default parameters is used, with
    the number of neurons derived from the encoder matrix dimensionality.
    use_jbias: if True, uses an external current source for the bias,
    otherwise decodes the bias from the pre-synaptic population.
    use_factorised_weights: if True, factorises the internal weight matrix into
    artificial encoders and decoders.
    """

    # Make sure the input arrays are correct
    assert len(encoders) > 0
    assert len(decoders) == len(encoders) == len(direct)

    # Fetch the number of neurons from the encoder dimensionality
    for i in range(1, len(encoders)):
        assert encoders[i].shape[0] == encoders[i - 1].shape[0]
    n_neurons = encoders[0].shape[0]

    # Make sure the dimensions in the input are correct
    for i in range(len(decoders)):
        assert encoders[i].shape[0] == scale_E.shape[0] == scale_I.shape[0]
        assert encoders[i].shape[1] == decoders[i].shape[0]

    def factorise_weights(weights):
        # Perform a SVD of the weight matrix
        U, S, V = np.linalg.svd(weights, full_matrices=False)

        # Calculate the number of dimensions needed to
        # represent the weight matrix
        k = np.sum(S > 1e-4 * np.max(S))

        # Return the encoder/decoder pair which approximates the input matrix
        E, D = (U @ np.diag(S))[:, 0:k], V[0:k, :]

        # Make sure the factorisation worked properly
        assert (
            np.max(np.abs(E @ D - weights)) < 1e-3 * np.max(np.abs(weights)))
        return E, D

    def decode_bias(bias, activities):
        # Assemble a single activity matrix from the activities
        m = max(map(lambda a: a.shape[0], activities))
        n = sum(map(lambda a: a.shape[1], activities))
        A = np.zeros((m, n))
        cur_n = 0
        for activity in activities:
            # Fetch the number of samples (m) and the number of neurons (n) for
            # the current activity matrix
            lm, ln = activity.shape

            # Repeat the matrix "rep" times, so each activity matrix has an
            # equal number of samples, assign the corresponding matrix to the
            # A matrix
            rep = int(math.ceil(m / lm))
            A[:, cur_n:(cur_n + ln)] = np.tile(activity, (rep, 1))[0:m, :]
            cur_n += ln

        # Desired output function Y -- just repeat "bias" m times
        Y = np.tile(bias, (m, 1))

        # Regularisation matrix
        I = np.eye(n) * ((np.max(A) * 0.1)**2)

        # Calculate the decoders using a least squares estimate
        return (np.linalg.inv(A.T @ A / m + I) @ A.T @ Y).T / m

    # Calculate the weight matrix for each input independently
    weights = [None] * len(decoders)
    direct_weights = [None] * len(decoders)
    n_in_total = len(decoders)
    n_in_direct = 0
    for i in range(len(decoders)):
        W = encoders[i] @ decoders[i]
        if direct[i]:  # Silence direct connections
            direct_weights[i] = W
            weights[i] = np.zeros(W.shape)
            n_direct += 1
        else:
            direct_weights[i] = np.zeros(W.shape)
            weights[i] = W
    weights = np.concatenate(weights, axis=1)
    direct_weights = np.concatenate(direct_weights, axis=1)

    voltage = np.zeros(n_neurons)
    refractory_time = np.zeros(n_neurons)
    spiked = np.zeros(n_neurons)

    def simulator(t, A):
        gE = np.maximum(0, scale_E * (WgE @ A + n_direct / n_in_total + direct_weights @ A))
        gI = np.maximum(0, scale_I * (WgI @ A + n_direct / n_in_total - direct_weights @ A))

        # Call the LIF neuron model with the calculated J
        spiked[:] = 0
        model.step_math(dt, gE, gI, spiked, voltage, refractory_time)

        return spiked

    return simulator


def extract_neuron_parameters(ens, sim):
    """
    Extracts x-intercept and max_rate from a neuron ensemble based on its gain
    and bias (assuming the ensemble is a LIF ensemble). Returns two vectors
    x_intercept and max_rate.

    ens: ensemble for which x_intercept and max_rate should be extracted.
    sim: simulation object containing the final bias and gain values.
    """
    bias = sim.data[ens].bias
    gain = sim.data[ens].gain

    tau_ref = ens.neuron_type.tau_ref
    tau_rc = ens.neuron_type.tau_rc
    max_rate = 1 / (tau_ref - tau_rc * np.log(1 - 1 / (bias + gain)))
    x_intercept = (1 - bias) / gain

    return x_intercept, max_rate


def calculate_conductance_neuron_parameters(x_intercept,
                                            max_rate,
                                            tau_ref=2e-3,
                                            e_rev_E=4.33,
                                            e_rev_I=-0.33):
    """
    Calculates the parameters gL, scale_E, and scale_I for a conductance
    based neuron under the assumption that GE(x) = 1 + x and GI(x) = 1 - x.

    x_intercept: vector describing the x_intercepts of inidividual neurons.
    max_rate: vector containing the maximum rates of the neurons.
    tau_ref: refractory period.
    e_rev_E: excitatory reversal potential.
    e_rev_I: inhibitory reversal potential.
    """

    # Make sure the input are two arrays of the same length
    max_rate = np.atleast_1d(max_rate).flatten()
    x_intercept = np.atleast_1d(x_intercept).flatten()
    assert (max_rate.shape[0] == x_intercept.shape[0])

    EE = e_rev_E
    EI = e_rev_I
    XI = x_intercept
    GL = -np.log((
        (-EE * XI + 2 * EE + XI - 2) / (3 * EE))) * (EE * XI + EE - XI - 1) / (
            (1 / max_rate - tau_ref) * (EE * XI + EE - XI + 2))
    scale_E = (3 * GL) / (2 * (EE * XI + EE - XI - 1))
    scale_I = (1 * GL) / (2 * (EI * XI - EI - XI + 1))

    return (GL, scale_E, scale_I)


def lif_cond_rate(GL, GE, GI, tau_ref=2e-3, EE=4.33, EI=-0.33, EL=0.0):
    G_tot = GL + GE + GI
    e_eq = (GL * EL + GE * EE + GI * EI) / G_tot

    mask = e_eq > 1
    exponent = mask * (e_eq - 1) / (mask * e_eq + (1 - mask))
    mask = exponent > 1e-15
    t_spike = -np.log(mask * exponent + (1 - mask)) / G_tot
    return mask / (tau_ref + t_spike)


def solve_for_nonneg_weigths(offs, scale, A, Y, E, sigma):
    from scipy.optimize import nnls

    Y = (offs + scale * (E @ Y)).clip(0, None)
    print("Y", Y.shape)
    print("A", A.shape)
    print("E", E.shape)
    print("sigma", sigma)


#    d = Y.shape[1]

#    GA = A.T @ A
#    np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma**2)
#    GY = A.T @ Y

#    X = np.zeros((n, d))


def get_activities(sim,
                   ens,
                   eval_points,
                   is_conductance_ensemble=False,
                   e_rev_E=4.33,
                   e_rev_I=-0.33):
    import nengo.builder.ensemble

    GL = None

    if is_conductance_ensemble:
        # Fetch the original x_intercept and max_rate for the pre population
        # and calculate the corresponding parameters for a conductance based
        # population.
        x_intercept, max_rate = extract_neuron_parameters(ens, sim)
        tau_ref = ens.neuron_type.tau_ref
        GL = 1 / ens.neuron_type.tau_rc

        scale_E, scale_I, bias_E, bias_I = lif_cond_utils.optimize_scale_E_scale_I_bias_E_bias_I(
            x_intercept, max_rate, gL=50, e_rev_E=e_rev_E, e_rev_I=e_rev_I, tau_ref=tau_ref)

        # Multiply the evaluation points with the encoders in order to get some
        # scalar values
        EX = eval_points @ sim.data[ens].encoders.T

        # Calculate the activities of the pre-population
        GE = np.maximum(0, scale_E * (bias_E + EX))
        GI = np.maximum(0, scale_I * (bias_I - EX))
        A = lif_cond_rate(GL, GE, GI, tau_ref, e_rev_E, e_rev_I)
    else:
        # Otherwise, if the pre-population is just a normal lif population, use
        # the normal activities function
        A = nengo.builder.ensemble.get_activities(sim.data[ens], ens,
                                                  eval_points)

    return A


def calculate_weights(sim,
                      conn,
                      pre_is_cond,
                      post_is_cond,
                      n_post_in,
                      e_rev_E=4.33,
                      e_rev_I=-0.33):
    """
    Used internally to calculates the weights of a connection. Takes different
    tuning curves for current and conductance based ensembles into account.

    sim: simulation object.
    conn: connection object.
    pre_is_cond: True if the pre-population is a conductance based ensemble.
    post_is_cond: True if the post-population is a conductance based ensemble.
    If set to True, this function will return two weight matrices, the positive
    weight matrix w_pos and the negative weight matrix w_neg.
    n_post_in: number of inputs for the post-synaptic ensemble. Only relevant if
    post_is_cond is True.
    e_rev_E, e_rev_I: excitatory and inhibitory reversal potential of the pre-
    and post-synaptic population. Only relevant if either of them is a
    conductance based population.
    """

    # Fetch the pre- and post-object
    ens_pre = conn.pre_obj
    ens_post = conn.post_obj

    # Make sure the pre-population object is an ensemble
    # TODO: Should work with anything.
    assert (isinstance(ens_pre, nengo.Ensemble))

    # Fetch the original evaluation points and corresponding activities
    eval_points = sim.data[ens_pre].eval_points
    A = get_activities(sim, ens_pre, eval_points, pre_is_cond)

    # Get the target values
    from nengo.build.connection import get_targets
    targets = get_targets(conn, eval_points)

    # Solve for the excitatory and the inhibitory weights
    solver = nengo.solvers.NnlsL2(weights=True, reg=0.1)
    w_pos = solve_for_nonneg_weigths(1 / n_post_in, 1, A, Y, E,
                                     np.max(A) * 0.1)
    w_neg = solve_for_nonneg_weigths(1 / n_post_in, -1, A, Y, E,
                                     np.max(A) * 0.1)


def transform_ensemble(
        ens,
        conn_ins,
        sim,
        e_rev_E=4.33,  # equiv. to 0mV for v_rest=-65mV, v_th=-50mV
        e_rev_I=-0.33,  # equiv. to -70mV
        use_linear_avg_pot=False,
        use_conductance_synapses=True,
        use_factorised_weights=False,
        use_jbias=False):
    """
    Creates an equivalent conductance based ensemble for the given input
    ensemble. Returns the node corresponding to the ensemble or None if the
    ensemble cannot be transformed. As a second return value returns a list
    containing the target dimensionalities of the newly created node for each
    dimension.

    ens: Ensemble that should be converted.
    conn_ins: list of input connections to that ensemble.
    conn_outs: list of output connections to that ensemble.
    sim: Simulation object that is used to fetch the decoders and encoders for
    the network.
    use_linear_avg_pot: if True, uses the linear average membrane potential
    estimate.
    use_conductance_synapses: if True, implements "normal" non-conductance based
    synapses. This is only useful for testing purposes.
    use_factorised_weights: if True, factorises the internal weight matrix into
    artificial encoders and decoders.
    use_jbias: if True, uses an external current source for the bias,
    otherwise decodes the bias from the pre-synaptic population.
    """

    # Make sure the ensemble this transformation operates on has either the
    # neuron type LIF or LIFRate. Fetch gains, biases, and encoders from the
    # ensemble.
    if not isinstance(ens.neuron_type, nengo.neurons.LIF):
        return None, None

    # Abort if the ensemble has no input.
    if len(conn_ins) == 0:
        return None, None

    # Abort if the user requested the ensemble not to be transformed
    if hasattr(
            ens,
            'use_conductance_synapses') and not ens.use_conductance_synapses:
        return None, None

    extract_neuron_parameters(ens, sim)

    n_neurons = ens.n_neurons
    encoder = sim.data[ens].encoders
    bias = sim.data[ens].bias
    gain = sim.data[ens].gain

    # Iterate over all input connections and calculate the total number of
    # neurons feeding into the node. Make sure the input connections are all
    # Ensemble objects. Fetch the decoders for the individual connections.
    n_dims_in = 0
    decoders = [None] * len(conn_ins)
    activities = [None] * len(conn_ins)
    encoders = [None] * len(conn_ins)
    connectivity = [None] * len(conn_ins)
    direct = [False] * len(conn_ins)
    for i, conn_in in enumerate(conn_ins):
        pre_obj = conn_in.pre_obj
        post_obj = conn_in.post_obj

        # Fetch the decoders and the number of neurons/dimensions in the pre-
        # ensemble
        if isinstance(pre_obj, nengo.Ensemble):
            decoders[i] = sim.data[conn_in].weights
            n_dims = pre_obj.n_neurons
        elif isinstance(pre_obj, nengo.ensemble.Neurons):
            pre_obj = pre_obj.ensemble
            n_dims = pre_obj.n_neurons
            W = sim.data[conn_in].weights
            if (np.ndim(W) == 0):
                W = np.eye(n_dims) * W
            elif (np.ndim(W) == 1):
                W = np.diag(W)
            out_dim, in_dim = W.shape
            decoders[i] = np.zeros((out_dim, n_dims))
            decoders[i][:, conn_in.pre_slice] = W
        elif isinstance(pre_obj, nengo.Node):
            n_dims = pre_obj.size_out
            decoders[i] = sim.data[conn_in].weights
            if (np.ndim(decoders[i]) == 0):
                decoders[i] = (
                    np.eye(n_dims) * decoders[i])[conn_in.pre_slice, :]
            direct[i] = True

        # Fetch the activities required for bias decoding
        if not use_jbias:
            if isinstance(pre_obj, nengo.Ensemble):
                activities[i] = get_activities(sim, pre_obj,
                                               sim.data[pre_obj].eval_points)
            elif isinstance(pre_obj, nengo.Node):
                activities[i] = np.zeros((1, pre_obj.size_out))

        # Apply the post-slice (pre-slice is already included in the decoder),
        # special treatment required for ".neurons" connections
        if isinstance(post_obj, nengo.ensemble.Neurons):
            encoders[i] = (
                np.eye(n_neurons) / gain.reshape(-1, 1))[:, conn_in.post_slice]
        else:
            encoders[i] = encoder[:, conn_in.post_slice]

        # Scale the encoders by the radius
        encoders[i] = encoders[i] / ens.radius

        connectivity[i] = slice(n_dims_in, n_dims_in + n_dims)
        n_dims_in += n_dims

    # Create the IfCondExp instance
    x_intercept, max_rate = extract_neuron_parameters(ens, sim)
    gL, scale_E, scale_I = calculate_conductance_neuron_parameters(
        x_intercept,
        max_rate,
        ens.neuron_type.tau_ref,
        e_rev_E=e_rev_E,
        e_rev_I=e_rev_I)
    model = lif_cond.LifCond(
        gL=gL,
        tau_ref=ens.neuron_type.tau_ref,
        e_rev_E=e_rev_E,
        e_rev_I=e_rev_I,
        #        use_linear_avg_pot=use_linear_avg_pot,
        #        use_conductance_synapses=use_conductance_synapses
    )

    # Assemble the simulator node
    node = nengo.Node(
        size_out=n_neurons,
        size_in=n_dims_in,
        output=sim_if_cond_exp(
            decoders=decoders,
            activities=activities,
            encoders=encoders,
            direct=direct,
            scale_E=scale_E,
            scale_I=scale_I,
            model=model,
            dt=sim.dt,
            use_jbias=use_jbias,
            use_factorised_weights=use_factorised_weights),
        label=ens.label)

    return node, connectivity


def transform(
        network_src,
        dt=None,
        e_rev_E=4.33,  # equiv. to 0mV for v_rest=-65mV, v_th=-50mV
        e_rev_I=-0.33,  # equiv. to -70mV
        use_linear_avg_pot=False,
        use_conductance_synapses=True,
        use_factorised_weights=False,
        use_jbias=False,
        seed=None):
    """
    Transforms the network network_src into an equivalent network model in which
    all LIF ensembles are replaced by LIF ensembles with conductance based
    inhibitory or excitatory synapses.

    network_src: source network that should be transformed.
    dt: timestep that should be used for the simulation of the new ensembles. If
    None, the dt of the given sim object is used.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    use_linear_avg_pot: if True, uses the linear average membrane potential
    estimate.
    use_conductance_synapses: if True, implements "normal" non-conductance based
    synapses. This is only useful for testing purposes.
    use_factorised_weights: if True, factorises the internal weight matrix into
    artificial encoders and decoders.
    use_jbias: if True, uses an external current source for the bias,
    otherwise decodes the bias from the pre-synaptic population.
    """

    rnd = np.random.RandomState(seed)

    def gen_seed():
        return rnd.randint(np.iinfo(np.int32).max)

    def fetch_pre_post(connection):
        pre_obj = connection.pre_obj
        post_obj = connection.post_obj
        if isinstance(pre_obj, nengo.ensemble.Neurons):
            pre_obj = pre_obj.ensemble
        if isinstance(post_obj, nengo.ensemble.Neurons):
            post_obj = post_obj.ensemble
        return pre_obj, post_obj

    # Collect the input and output connections for each ensemble
    ensemble_info = {}
    for ensemble in network_src.all_ensembles:
        ensemble_info[ensemble] = {
            'conn_ins': [],
            'conn_outs': [],
            'transformed': None
        }

    # Iterate over all connections in the network and fetch all connections
    # belonging to an ensemble
    for connection in network_src.all_connections:
        pre_obj, post_obj = fetch_pre_post(connection)

        # Generate seeds for all objects involved in the connection
        # which do not have an connection object yet
        if pre_obj.seed is None:
            pre_obj.seed = gen_seed()
        if post_obj.seed is None:
            post_obj.seed = gen_seed()
        if connection.seed is None:
            connection.seed = gen_seed()

        # Sort the connections into their corresponding buckets
        if (pre_obj in ensemble_info):
            ensemble_info[pre_obj]['conn_outs'].append(connection)
        if (post_obj in ensemble_info):
            ensemble_info[post_obj]['conn_ins'].append(connection)

    # Recursively rebuild all the network instances
    def transform_network(net_src, parent=None):
        with nengo.Network(
                label=net_src.label, seed=net_src.seed,
                add_to_container=parent) as net_tar:
            for ensemble_src in net_src.ensembles:
                # Try to convert the ensemble
                ensemble_tar, connectivity = transform_ensemble(
                    ensemble_src,
                    ensemble_info[ensemble_src]['conn_ins'],
                    sim,
                    e_rev_E=e_rev_E,
                    e_rev_I=e_rev_I,
                    use_linear_avg_pot=use_linear_avg_pot,
                    use_conductance_synapses=use_conductance_synapses,
                    use_factorised_weights=use_factorised_weights,
                    use_jbias=use_jbias)

                # If the conversion was not successful, just use the original
                # ensemble instead
                if ensemble_tar is None:
                    net_tar.add(ensemble_src)

                # Remember the mapping between old and new ensembles, store the
                # connectivity matrix
                ensemble_info[ensemble_src]['transformed'] = ensemble_tar
                ensemble_info[ensemble_src]['connectivity'] = connectivity

            # Add all old nodes to the new network
            for node in net_src.nodes:
                net_tar.add(node)

            # Descend into all subnetworks
            for subnet in net_src.networks:
                transform_network(subnet, net_tar)

        # Return the transformed network
        return net_tar

    # Create a simulator object with the given dt if None has been given.
    connection_translation = {}
    with nengo.simulator.Simulator(network_src, dt=dt) as sim, \
            transform_network(network_src) as network_tar:
        # Rebuild all connections in the network
        for connection in network_src.all_connections:
            pre_obj, post_obj = fetch_pre_post(connection)
            pre_obj_transformed = (
                (pre_obj in ensemble_info) and
                (not ensemble_info[pre_obj]['transformed'] is None))
            post_obj_transformed = (
                (post_obj in ensemble_info) and
                (not ensemble_info[post_obj]['transformed'] is None))

            if not (pre_obj_transformed or post_obj_transformed):
                connection_tar = connection
                network_tar.add(connection_tar)
            elif not pre_obj_transformed and post_obj_transformed:
                info = ensemble_info[post_obj]
                idx = info['conn_ins'].index(connection)
                src = pre_obj
                if isinstance(pre_obj, nengo.Ensemble):
                    src = pre_obj.neurons
                connection_tar = nengo.Connection(
                    src,
                    info['transformed'][info['connectivity'][idx]],
                    synapse=connection.synapse,
                    seed=connection.seed,
                    label=connection.label)
            elif pre_obj_transformed and not post_obj_transformed:
                info = ensemble_info[pre_obj]
                connection_tar = nengo.Connection(
                    info['transformed'],
                    post_obj,
                    transform=sim.data[connection].weights,
                    synapse=connection.synapse,
                    seed=connection.seed,
                    label=connection.label)
            elif pre_obj_transformed and post_obj_transformed:
                pre_info = ensemble_info[pre_obj]
                post_info = ensemble_info[post_obj]
                idx = post_info['conn_ins'].index(connection)
                connection_tar = nengo.Connection(
                    pre_info['transformed'],
                    post_info['transformed'][post_info['connectivity'][idx]],
                    synapse=connection.synapse,
                    seed=connection.seed,
                    label=connection.label)

    return network_tar

