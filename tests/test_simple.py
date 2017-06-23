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

import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nengo
#import nengo_conductance_synapses as conductance_synapses
import nengo_conductance_synapses.preprocess as preprocess
import numpy as np

T = 1.0
dt = 1e-3


def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))


def transform_and_test(test, model_src, tar):
    # Probe the source model
    with model_src:
        pout_src = nengo.Probe(tar)

    for ens in model_src.all_ensembles:
        ens.seed = 1

    # Transform the model
    model_trafo, probes = preprocess.preprocess(model_src)
    pout_trafo = probes[pout_src]

    # Run the source model
    with nengo.Simulator(model_src, dt=dt) as sim:
        sim.run(T)
        data_src = sim.data[pout_src]

    # Run the transformed model
    with nengo.Simulator(model_trafo, dt=dt) as sim:
        sim.run(T)
        data_trafo = np.array(pout_trafo.data) #sim.data[pout_trafo]

    # Make sure the traces are almost equal. Note that conductance based
    # synapses were deactivated when the transform function was called. Only
    # the network graph transformation itself is tested.
    test.assertAlmostEqual(rmse(data_src, data_trafo), 0.0)


class TestSimple(unittest.TestCase):
    def test_no_ensemble(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, tar)

        transform_and_test(self, model_src, tar)

    def test_single_ensemble(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens = nengo.Ensemble(100, 1, label="ens")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens)
            nengo.Connection(ens, tar)

        transform_and_test(self, model_src, tar)

    def test_single_ensemble_radius(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens = nengo.Ensemble(100, 1, radius=1.5, label="ens")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens)
            nengo.Connection(ens, tar)

        transform_and_test(self, model_src, tar)

    def test_single_2d_ensemble_slice(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens = nengo.Ensemble(100, 2, label="ens")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens[0], transform=0.3)
            nengo.Connection(src, ens[1], transform=0.7)
            nengo.Connection(ens[0], tar, transform=0.1)
            nengo.Connection(ens[1], tar, transform=0.4)

        transform_and_test(self, model_src, tar)

    def test_recurrent_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens = nengo.Ensemble(100, 1, label="ens")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens, transform=0.1)
            nengo.Connection(ens, ens, synapse=0.1)
            nengo.Connection(ens, tar, synapse=0.05)

        transform_and_test(self, model_src, tar)

    def test_single_ensemble_untransformed_nodes(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            src2 = nengo.Node(size_in=1, label="src2")
            ens = nengo.Ensemble(100, 1, label="ens")
            tar = nengo.Node(size_in=1, label="tar")
            tar2 = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, src2)
            nengo.Connection(src2, ens)
            nengo.Connection(ens, tar)
            nengo.Connection(tar, tar2)

        transform_and_test(self, model_src, tar2)

    def test_single_two_dimensional_ensemble(self):
        with nengo.Network() as model_src:
            src = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label="src")
            ens = nengo.Ensemble(100, 2, label="ens")
            tar = nengo.Node(size_in=2, label="tar")

            nengo.Connection(src, ens)
            nengo.Connection(ens, tar)

        transform_and_test(self, model_src, tar)

    def test_communication_channel(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2)
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_communication_channel_with_f(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2, function=lambda x: x**2)
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_communication_channel_with_2d_f(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 2, label="ens2")
            tar = nengo.Node(size_in=2, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2, function=lambda x: [x, x**2])
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_communication_channel_chain(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            ens3 = nengo.Ensemble(75, 1, label="ens3")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2)
            nengo.Connection(ens2, ens3)
            nengo.Connection(ens3, tar)

    def test_communication_channel_chain_with_trafos(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 3, label="ens2")
            ens3 = nengo.Ensemble(75, 2, label="ens3")
            ens4 = nengo.Ensemble(25, 1, label="ens4")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1, transform=2)
            nengo.Connection(ens1, ens2, transform=[[1], [0.5], [-0.2]])
            nengo.Connection(
                ens2, ens3, transform=[[0.5, 0.3, 0.1], [0.7, 0.5, 0.4]])
            nengo.Connection(ens3, ens4, transform=[[0.3, 0.2]])
            nengo.Connection(ens4, tar, transform=4)

        transform_and_test(self, model_src, tar)

    def test_communication_channel_chain_with_trafos_and_f(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 3, label="ens2")
            ens3 = nengo.Ensemble(75, 2, label="ens3")
            ens4 = nengo.Ensemble(25, 1, label="ens4")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2, transform=[[1], [0.5], [-0.2]])
            nengo.Connection(
                ens2,
                ens3,
                transform=[[0.5, 0.3], [0.7, 0.4]],
                function=lambda v: [(v[0] + v[1]) * v[2], v[2]])
            nengo.Connection(ens3, ens4, transform=[[0.3, 0.2]])
            nengo.Connection(ens4, tar)

        transform_and_test(self, model_src, tar)

    def test_communication_channel_chain_with_slices(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(100, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 3, label="ens2")
            ens3 = nengo.Ensemble(75, 2, label="ens3")
            ens4 = nengo.Ensemble(25, 1, label="ens4")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1, transform=2)
            nengo.Connection(ens1, ens2[0:1], transform=0.3)
            nengo.Connection(ens1, ens2[1], transform=0.2)
            nengo.Connection(
                ens2[0:2], ens3, transform=[[0.3, 0.1], [0.5, 0.4]])
            nengo.Connection(ens2[2], ens3[1], transform=0.4)
            nengo.Connection(ens3[0], ens4, transform=0.2)
            nengo.Connection(ens3[1], ens4, transform=0.3)
            nengo.Connection(ens4, tar, transform=4)

        transform_and_test(self, model_src, tar)

    def test_multiple_ensembles(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1a = nengo.Ensemble(50, 1, label="ens1a")
            ens1b = nengo.Ensemble(50, 1, label="ens1b")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1a, transform=0.3)
            nengo.Connection(src, ens1b, transform=0.7)
            nengo.Connection(ens1a, tar, transform=0.6)
            nengo.Connection(ens1b, tar, transform=0.4)

        transform_and_test(self, model_src, tar)

    def test_multiple_ensembles_with_2f(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1a = nengo.Ensemble(110, 1, label="ens1a")
            ens1b = nengo.Ensemble(120, 1, label="ens1b")
            ens2 = nengo.Ensemble(130, 3, label="ens2")
            tar = nengo.Node(size_in=3, label="tar")

            nengo.Connection(src, ens1a, transform=0.1)
            nengo.Connection(src, ens1b, transform=0.3)
            nengo.Connection(ens1a, ens2[0:2], function=lambda x: [x**2, x])
            nengo.Connection(ens1b, ens2[2], function=lambda x: x**3)
            nengo.Connection(
                ens2, ens2, function=lambda x: [x[0] - 1, x[1] + x[2], x[0]])
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)


    def test_multiple_ensembles_chained(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1a = nengo.Ensemble(50, 1, label="ens1a")
            ens1b = nengo.Ensemble(50, 1, label="ens1b")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            ens3a = nengo.Ensemble(50, 1, label="ens3a")
            ens3b = nengo.Ensemble(50, 1, label="ens3b")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1a, transform=0.3)
            nengo.Connection(src, ens1b, transform=0.7)
            nengo.Connection(ens1a, ens2, transform=0.6)
            nengo.Connection(ens1b, ens2, transform=0.4)
            nengo.Connection(ens2, ens3a, transform=1.2)
            nengo.Connection(ens2, ens3b, transform=0.8)
            nengo.Connection(ens1a, ens3b, transform=1.2)
            nengo.Connection(ens1b, ens3a, transform=1.3)
            nengo.Connection(ens3a, tar)
            nengo.Connection(ens3b, tar)

        transform_and_test(self, model_src, tar)

    def test_pre_neurons_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(60, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(
                ens1.neurons, ens2, transform=1e-4 * np.ones((1, 50)))
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_pre_neurons_slice_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(60, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1.neurons[5:20], ens2, transform=1e-3 * np.ones((1, 15)))
            nengo.Connection(ens2, tar)

    def test_post_neurons_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(60, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2.neurons, transform=np.ones((60, 1)))
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_post_neurons_slice_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(60, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1, ens2.neurons[10:20], transform=np.ones((10, 1)))
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_neurons_connection(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1.neurons, ens2.neurons)
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_neurons_connection_slice(self):
        with nengo.Network() as model_src:
            src = nengo.Node(np.sin, label="src")
            ens1 = nengo.Ensemble(50, 1, label="ens1")
            ens2 = nengo.Ensemble(50, 1, label="ens2")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src, ens1)
            nengo.Connection(ens1.neurons[10:20], ens2.neurons[20:30])
            nengo.Connection(ens2, tar)

        transform_and_test(self, model_src, tar)

    def test_2d_node_input(self):
        with nengo.Network() as model_src:
            src = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label="src")
            ens = nengo.Ensemble(50, 1, label="ens")
            tar = nengo.Node(size_in=1, label="tar")

            nengo.Connection(src[0], ens)
            nengo.Connection(src[1], ens)
            nengo.Connection(ens, tar)

        transform_and_test(self, model_src, tar)

if __name__ == '__main__':
    unittest.main()

