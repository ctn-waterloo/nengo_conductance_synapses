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

import nengo
import numpy as np

class LifCond:
    """
    The LifRateCond class represents a population of lif neurons with
    conductance based synapses. This class is used as storage for neuron and
    synapse parameters.
    """

    def __init__(
            self,
            gL=50.0,
            tau_ref=2.0e-3,
            e_rev_E=4.33,  # equiv. to 0mV for v_rest=-65mV, v_th=-50mV
            e_rev_I=-0.33  # equiv. to -70mV
    ):
        """
        Constructor of the LifRateCond neuron class. Copies the given parameters
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
        """

        # Copy all parameters
        self.gL = gL
        self.tau_ref = tau_ref
        self.e_rev_E = e_rev_E
        self.e_rev_I = e_rev_I

    def step_math(self, dt, gE, gI, spiked, voltage, refractory_time):
        # make sure gE, gI are subscriptable numpy arrays
        gE, gI = np.atleast_1d(gE), np.atleast_1d(gI)

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # calculate the equilibrium potential
        g_tot = (self.gL + gE + gI)
        e_eq = (gE * self.e_rev_E + gI * self.e_rev_I) / g_tot

        # update voltage assuming conductances are constant
        voltage[...] = (voltage - e_eq) * np.exp(-delta_t * g_tot) + e_eq

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[...] = spiked_mask / dt

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + np.log((e_eq[spiked_mask] - voltage[spiked_mask]) /
                              (e_eq[spiked_mask] - 1)) / g_tot[spiked_mask]

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < 0] = 0
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike

