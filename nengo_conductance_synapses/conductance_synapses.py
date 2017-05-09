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


class IfCondExp(nengo.LIF):
    """
    The IfCondExp class represents a population of IfCondExp neurons. This
    neuron type is a LIF neuron with conductance based synapses. This class is
    used as storage for neuron and synapse parameters, as well as the neuron and
    synapse state.
    """

    def __init__(
            self,
            n_neurons,
            tau_rc=20.0e-3,
            tau_ref=2.0e-3,
            e_rev_E=4.33,  # equiv. to 0mV for v_rest=-65mV, v_th=-50mV
            e_rev_I=-0.33,  # equiv. to -70mV
            use_linear_avg_pot=False,
            use_conductance_synapses=True):
        """
        Constructor of the IfCondExp neuron class. Copies the given parameters
        and instantiates the internal state vectors. State vectors hold the
        membrane potential, the refractory time, and the conductivity of the
        excitatory and inhibitory synapses for each individual neuron.

        n_neurons: number of neurons in the population.
        tau_rc: membrane time constant of the underlying LIF neuron
        tau_ref: refractory period of the underlying LIF neuron
        e_rev_E: reversal potential of the excitatory synapse. Potentials are
        normalised in such a way that a value zero corresponds to the resting
        potential and a value of one to the threshold potential.
        e_rev_I: reversal potential of the inhibitory synapse.
        use_linear_avg_pot: if True, estimates the average membrane potential
        using a simple linear estimation. If False, takes the non-linearity in
        the transition between 0 and 1 into account.
        use_conductance_synapses: if False, reduced the IfCondExp model to the
        standard LIF model with current based synapses. This flag is useful for
        testing the implementation.
        """
        # Instantiate the base class
        super(IfCondExp, self).__init__(tau_rc=tau_rc, tau_ref=tau_ref)

        # Copy all other parameters
        self.n_neurons = n_neurons
        self.e_rev_E = e_rev_E
        self.e_rev_I = e_rev_I
        self.use_conductance_synapses = use_conductance_synapses

        # Instantiate the state vectors
        self.g_E = np.zeros(n_neurons)
        self.g_I = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)
        self.voltage = np.zeros(n_neurons)

        # Calculate the scale factors
        EV = IfCondExp.average_membrane_potential_estimate(
            e_rev_E, use_linear_avg_pot=use_linear_avg_pot)
        self.scale_E = 1 / (self.e_rev_E - EV)
        self.scale_I = 1 / (-self.e_rev_I + EV)

    @staticmethod
    def average_membrane_potential_estimate(e_rev_E, use_linear_avg_pot=False):
        """
        Function which estimates the average membrane potential under the
        assumption of a high output rate.

        e_rev_E: excitatory synapse reversal potential.
        use_linear_avg_pot: if True, returns 0.5, which is the average when
        assuming a linear membrane potential transition between the resting
        potential zero and the threshold potential 1.
        """
        if use_linear_avg_pot:
            return 0.5
        alpha = math.log((e_rev_E - 1) / e_rev_E)
        return (alpha * e_rev_E + 1) / alpha

    def update_synapses(self, A, E_pos, E_neg, D_pos=None, D_neg=None):
        """
        Computes the current that is being injected into the membrane. The
        current depends on the synaptic conductivities and the membrane
        potential.

        dt: simulation timestep to be used.
        A: pre synaptic neuron population activity vector.
        E_pos: encoders for the positive coefficients in the weight matrix. If
        D_pos is not specified (is None), this parameter defines the positive
        weight matrix w_pos instead.
        E_neg: encoders for the negative coefficients in the weight matrix. If
        D_neg is not specified (is None), this parameter defines the negative
        weight matrix w_neg instead.
        D_pos: decoders for the positive coefficients in the weight matrix.
        D_neg: decoders for the negative coefficients in the weight matrix.
        """

        # Process incomming spikes by incrementing the conductivity.
        # Note: the scaling factor ensures a unit-area below the synaptic
        # filter.
        if (D_pos is None) or (D_neg is None):
            # In this branch, E_pos and E_neg are actually w_pos and w_neg
            self.g_E = E_pos @ A
            self.g_I = E_neg @ A
        else:
            self.g_E = E_pos @ (D_pos @ A)
            self.g_I = E_neg @ (D_neg @ A)

        # Calculate the current that is being induced into the membrane.
        # In an actual implementation, the scaling factors can be
        # multiplied into the synaptic weights.
        J = (self.scale_E * self.g_E *
             (self.e_rev_E - self.voltage) + self.scale_I * self.g_I *
             (self.e_rev_I - self.voltage))

        # When using this equation, the results must be equivalent to
        # the LIF model with current based synapse
        if not self.use_conductance_synapses:
            J = self.g_E - self.g_I

        return J


def sim_if_cond_exp(decoders,
                    activities,
                    encoders,
                    bias,
                    gain,
                    model,
                    dt=1e-4,
                    use_jbias=False,
                    use_factorised_weights=True):
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
    bias: bias values that should be used for this population.
    gain: gain values that should be used for this population.
    model: instance of the IfCondExp class defined above. If None
    is given, a new instance with default parameters is used, with
    the number of neurons derived from the encoder matrix dimensionality.
    use_jbias: if True, uses an external current source for the bias,
    otherwise decodes the bias from the pre-synaptic population.
    use_factorised_weights: if True, factorises the internal weight matrix into
    artificial encoders and decoders.
    """

    # Fetch the number of neurons from the encoder dimensionality
    n_neurons = model.n_neurons

    # Make sure the dimensions in the input are correct
    assert (len(decoders) == len(encoders))
    for i in range(len(decoders)):
        assert (encoders[i].shape[0] == bias.shape[0] == gain.shape[0] ==
                model.n_neurons)
        assert (encoders[i].shape[1] == decoders[i].shape[0])

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
    for i in range(len(decoders)):
        weights[i] = (encoders[i] * gain.reshape(-1, 1)) @ decoders[i]
    weights = np.concatenate(weights, axis=1)

    # Split the weight matrix into the positive and negative part
    if not use_jbias:
        try:
            weights += decode_bias(bias, activities)
        except np.linalg.linalg.LinAlgError:
            use_jbias = True  # The activity matrix is singular
            pass
    w_pos = weights * (weights > 0)
    w_neg = -weights * (weights < 0)

    # Factorise the weight matrices into positive and negative encoders/
    # decoders (this is not really required, but it speeds up the following
    # computations)
    if use_factorised_weights:
        E_pos, D_pos = factorise_weights(w_pos)
        E_neg, D_neg = factorise_weights(w_neg)

    def simulator(t, A):
        # Calculate the current current induced by the conductance based
        # synapses
        if use_factorised_weights:
            J = model.update_synapses(A, E_pos, E_neg, D_pos, D_neg)
        else:
            J = model.update_synapses(A, w_pos, w_neg)

        # If jbias is used instead of the decoding, add it to the neurons
        if use_jbias:
            J += bias

        # Call the LIF neuron model with the calculated J
        spiked = np.zeros(n_neurons)
        model.step_math(dt, J, spiked, model.voltage, model.refractory_time)

        return spiked

    return simulator


def transform_ensemble(
        ens,
        conn_ins,
        sim,
        e_rev_E=4.33,  # equiv. to 0mV for v_rest=-65mV, v_th=-50mV
        e_rev_I=-0.33,  # equiv. to -70mV
        use_linear_avg_pot=False,
        use_conductance_synapses=True,
        use_factorised_weights=True,
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

    from nengo.builder.ensemble import get_activities

    # Make sure the ensemble this transformation operates on has either the
    # neuron type LIF or LIFRate. Fetch gains, biases, and encoders from the
    # ensemble.
    if not isinstance(ens.neuron_type, nengo.neurons.LIF):
        return None, None

    # Abort if the ensemble has no input.
    if len(conn_ins) == 0:
        return None, None

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
                decoders[i] = np.eye(n_dims) * decoders[i]

        # Fetch the activities required for bias decoding
        if not use_jbias:
            if isinstance(pre_obj, nengo.Ensemble):
                activities[i] = get_activities(sim.data[pre_obj], pre_obj,
                                               sim.data[pre_obj].eval_points)
            elif isinstance(pre_obj, nengo.Node):
                activities[i] = np.zeros((1, pre_obj.size_out))

        # Apply the post-slice (pre-slice is already included in the decoder),
        # special treatment required for ".neurons" connections
        if isinstance(post_obj, nengo.ensemble.Neurons):
            encoders[i] = (np.eye(n_neurons) / gain.reshape(-1, 1))[:, conn_in.post_slice]
        else:
            encoders[i] = encoder[:, conn_in.post_slice]

        # Scale the encoders by the radius
        encoders[i] = encoders[i] / ens.radius

        connectivity[i] = list(range(n_dims_in, n_dims_in + n_dims))
        n_dims_in += n_dims

    # Create the IfCondExp instance
    model = IfCondExp(
        ens.n_neurons,
        tau_rc=ens.neuron_type.tau_rc,
        tau_ref=ens.neuron_type.tau_ref,
        e_rev_E=e_rev_E,
        e_rev_I=e_rev_I,
        use_linear_avg_pot=use_linear_avg_pot,
        use_conductance_synapses=use_conductance_synapses)

    # Assemble the simulator node
    node = nengo.Node(
        size_out=n_neurons,
        size_in=n_dims_in,
        output=sim_if_cond_exp(
            decoders=decoders,
            activities=activities,
            encoders=encoders,
            bias=bias,
            gain=gain,
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
        use_factorised_weights=True,
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
    with nengo.Simulator(network_src, dt=dt) as sim, \
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

