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
import nengo_conductance_synapses as conductance_synapses
import numpy as np

from test_simple import transform_and_test

T = 1.0
dt = 1e-4


class TestOscillator(unittest.TestCase):
    # Benchmark code adapted from https://github.com/tcstewar/nengo_benchmarks
    f_max = 2.0
    n_freq = 3
    N_osc = 100
    N_control = 100
    T = 1.0
    pstc = 0.01

    def test_oscillator(self):
        stims = np.linspace(-1, 1, self.n_freq)

        model = nengo.Network()
        with model:
            state = nengo.Ensemble(
                n_neurons=self.N_osc, dimensions=3, radius=1.7, label="state")

            def feedback(x):
                x0, x1, f = x
                w = f * self.f_max * 2 * np.pi
                return x0 + w * self.pstc * x1, x1 - w * self.pstc * x0

            nengo.Connection(
                state, state[:2], function=feedback, synapse=self.pstc)

            freq = nengo.Ensemble(
                n_neurons=self.N_control, dimensions=1, label="freq")
            nengo.Connection(freq, state[2], synapse=self.pstc)

            stim = nengo.Node(lambda t: [1, 0, 0] if t < 0.08 else [0, 0, 0])
            nengo.Connection(stim, state)

            def control(t):
                index = int(t / self.T) % self.n_freq
                return stims[index]

            freq_control = nengo.Node(control)

            nengo.Connection(freq_control, freq)

            out_state = nengo.Node(size_in=3, label="out_state")
            out_freq = nengo.Node(size_in=1, label="out_freq")
            nengo.Connection(state, out_state, synapse=None)
            nengo.Connection(freq, out_freq, synapse=None)

            p_state = nengo.Probe(out_state, synapse=0.03)
            p_freq = nengo.Probe(out_freq, synapse=0.03)

        transform_and_test(self, model, out_state)


if __name__ == '__main__':
    unittest.main()

