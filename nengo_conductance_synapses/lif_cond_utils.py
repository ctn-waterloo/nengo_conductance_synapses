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

import numpy as np


def _rate(g_tot, e_eq, tau_ref=2e-3, v_th=1.0):
    """
    Function used internally by lif_cond_rate and lif_rate. Takes the total
    conductance and the current equilibrium potential and calculates the
    corresponding spike rate. All parameters may be numpy arrays.

    g_tot: total conductance.
    e_eq: equilibrium potential.
    tau_ref: refractory period in second.
    v_th: threshold potential.
    """

    mask = 1 * (e_eq > v_th)  # Mask out invalid values
    t_spike = -np.log1p(-mask * v_th / (e_eq + (1 - mask))) / g_tot
    return mask / (tau_ref + t_spike)


def lif_cond_rate(gL,
                  gE,
                  gI,
                  e_rev_E=4.33,
                  e_rev_I=-0.33,
                  tau_ref=2e-3,
                  v_th=1.0):
    """
    Calculates the firing rate of a LIF neuron with conductance based synapses.
    Input to the model are conductances gE and gI. All parameters may be numpy
    arrays.

    gL: leak conductance.
    gE: excitatory conductance.
    gI: inhibitory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    v_th: threshold potential.
    """

    # Clamp gE, gI to non-negative values
    gE, gI = np.maximum(0, gE), np.maximum(0, gI)

    # Calculate the total conductance and the equilibrium potential
    g_tot = gL + gE + gI
    e_eq = (gE * e_rev_E + gI * e_rev_I) / g_tot

    return _rate(g_tot, e_eq, tau_ref, v_th)


def lif_rate(gL, J, tau_ref=2e-3, v_th=1.0):
    """
    Calculates the firing rate of a LIF neuron with conductance based synapses.
    Input to the model is conductances gE and gI. All parameters may be numpy
    arrays.

    gL: leak conductance.
    gE: excitatory conductance.
    gI: inhibitory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    v_th: threshold potential.
    """

    e_eq = J / gL
    return _rate(gL, e_eq, tau_ref, v_th)


def calc_gE_for_rate(rate,
                     gL,
                     gI,
                     e_rev_E=4.33,
                     e_rev_I=-0.33,
                     tau_ref=2e-3,
                     atol=1e-9):
    """
    Calculates the excitatory conductivity required to achive a certain spike
    rate given an inhibitory conductivity.

    gL: leak conductance.
    gI: inhibitory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    """

    # Convert the rate to a time-to-spike s
    s = 1 / rate - tau_ref

    # Shorthands
    EE = e_rev_E
    EI = e_rev_I
    gIs = np.atleast_1d(gI)

    # Solve t_spike_cond for gI using Newton's method
    x = np.ones(gIs.shape) * 1000
    for i, gI in enumerate(gIs):
        step = 1
        while np.abs(step) > atol:
            e = np.exp(-s * (gL + gI + x[i]))
            f = (x[i] * EE + gI * EI) * e - x[i] * \
                EE - gI * EI + x[i] + gI + gL
            df = (-x[i] * EE * s - gI * EI * s + EE) * e - EE + 1

            step = f / df
            x[i] = x[i] - step
    return x


def calc_gI_for_rate(rate, gL, gE, e_rev_E=4.33, e_rev_I=-0.33, tau_ref=2e-3):
    """
    Calculates the inhibitory conductivity required to achive a certain spike
    rate given an inhibitory conductivity.

    gL: leak conductance.
    gE: excitatory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    """

    # Just swap e_rev_E and e_rev_I and call the previous function.
    return calc_gE_for_rate(rate, gL, gE, e_rev_I, e_rev_E, tau_ref)


def calc_gE_for_intercept(gL, gI, e_rev_E=4.33, e_rev_I=-0.33):
    """
    Calculates the excitatory conductivity required to
    """
    return (gI * e_rev_I - gI - gL) / (1 - e_rev_E)


def calc_gL_scale_E_scale_I(x_intercept,
                            max_rate,
                            e_rev_E=4.33,
                            e_rev_I=-0.33,
                            tau_ref=2e-3):
    """
    Calculates the leak conductance, scale_E, and scale_I for x_intercept and
    max_rate.
    """
    EE = e_rev_E
    EI = e_rev_I
    XI = x_intercept
    GL = -np.log((
        (-EE * XI + 2 * EE + XI - 2) / (3 * EE))) * (EE * XI + EE - XI - 1) / (
            (1 / max_rate - tau_ref) * (EE * XI + EE - XI + 2))
    scale_E = (3 * GL) / (2 * (EE * XI + EE - XI - 1))
    scale_I = (1 * GL) / (2 * (EI * XI - EI - XI + 1))

    return (GL, scale_E, scale_I)


def calc_optimal_a_b_c_d(x_intercept,
                         max_rate,
                         gL,
                         e_rev_E=4.33,
                         e_rev_I=-0.33,
                         tau_ref=2e-3,
                         fan_in=1):
    """
    Calculates the optimal gains and biases for the affine gE(x) and gI(x)
    equations

        gE(x) = a * x + b
        gI(x) = c * x + d

    for the given x_intercept and max_rate.
    """

    EE = e_rev_E
    EI = e_rev_I
    xi = x_intercept
    n = fan_in

    def solve_for_gEoffs(gEoffs):
        alpha = -(EI - 1) / (EE - 1)
        a0 = -gEoffs / (xi - 1)
        b0 = (EE * gEoffs * xi - gEoffs * xi + gL * xi - gL) / (
            (EE - 1) * (xi - 1))

        solutions = np.array([
            [0, 0],
            [-(a0 * n + b0) / (2 * alpha * n), -(a0 * n + b0) / (2 * alpha)],
            [-(a0 * n - b0) / (2 * alpha * n), (a0 * n - b0) / (2 * alpha)],
            [-a0 / alpha, -b0 / alpha],
        ])

        c, d = solutions[np.argmax(solutions[:, 1]), :]
        a = alpha * c + a0
        b = alpha * d + b0

        return a, b, c, d

    def calc_gEoffs(gE, gI):
        return gE - (gL - gI * (EI - 1)) / (EE - 1)

    gEoffs = np.inf
    gEoffsNew = calc_gEoffs(
        calc_gE_for_rate(max_rate, gL, 0, e_rev_E, e_rev_I, tau_ref), 0)
    while np.abs(gEoffs - gEoffsNew) > 1e-3:
        gEoffs = gEoffsNew
        a, b, c, d = solve_for_gEoffs(gEoffs)

        gI = c + d
        gE = calc_gE_for_rate(max_rate, gL, gI, e_rev_E, e_rev_I)
        gEoffsNew = calc_gEoffs(gE, gI)

    return a, b, c, d


def solve_max_rate_x_intercept(x_intercept,
                               max_rate,
                               gL=None,
                               e_rev_E=4.33,
                               e_rev_I=-0.33,
                               tau_ref=2e-3,
                               v_th=1.0,
                               fan_in=1):
    """
    Calculates scale and bias factors for gL, gE, and gI for a conductance based
    synapse tuning curve with maximum rate max_rate and x-intercept x_intercept.

    max_rate: maximum rate of the tuning curve.
    x_intercept: x-intercept of the tuning curve.
    gL: if None, calculates a matching gL. Otherwise, keeps gL constant.
    """

    if gL is None:
        gL, scale_E, scale_I = calc_gL_scale_E_scale_I(
            x_intercept, max_rate, e_rev_E, e_rev_I, tau_ref)
        a, b = scale_E, scale_E
        c, d = scale_I, scale_I
    else:
        a, b, c, d = calc_optimal_a_b_c_d(x_intercept, max_rate, gL, e_rev_E,
                                          e_rev_I, tau_ref, fan_in)

    return gL, a, b, c, d

