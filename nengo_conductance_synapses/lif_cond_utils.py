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


def lif_cond_calc_g_tot_e_eq(gL, gE, gI, e_rev_E=4.33, e_rev_I=-0.33):
    g_tot = gL + gE + gI
    e_eq = (gE * e_rev_E + gI * e_rev_I) / g_tot
    return g_tot, e_eq


def lif_cond_rate(gL, gE, gI, e_rev_E=4.33, e_rev_I=-0.33,
                  tau_ref=2e-3, v_th=1.0):
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

    # Calculate the equilibrium potential
    g_tot, e_eq = lif_cond_calc_g_tot_e_eq(gL, gE, gI, e_rev_E, e_rev_I)

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


def optimize_scale_E_scale_I_bias_E_bias_I(
        x_intercept, max_rate, gL=50, e_rev_E=4.33, e_rev_I=-0.33, tau_ref=2e-3):
    """
    Optimizes scale_E, scale_I, bias_E, and bias_I for the given x_intercept and
    max_rate.
    """

    import scipy.optimize

    x_intercept = np.atleast_1d(x_intercept)
    max_rate = np.atleast_1d(max_rate)
    tau_ref = np.atleast_1d(tau_ref)
    assert len(x_intercept) == len(
        max_rate), "x_intercept and max_rate must have the same length"

    N = len(x_intercept)
    EE = e_rev_E
    EI = e_rev_I

    t_spike_tar = 1 / max_rate - tau_ref
    assert np.all(
        t_spike_tar > 0), "max_rate must be smaller than the inverse of tau_ref"

    def optimize_single(xi, t_tar):
        def calc_a(b, c, d):
            return - (b * (EI * d - EI * xi - d + xi) - gL) / \
                (EE * c + EE * xi - c - xi)

        def f(v):
            b, c, d = v
            a = calc_a(b, c, d)

            gE = a * (c + 1)
            gI = b * (d - 1)

            g_tot = gL + gE + gI
            e_eq = (gE * EE + gI * EI) / g_tot

            t = 0
            if e_eq > 1:
                t = -np.log1p(-1 / e_eq) / g_tot

            lambda_ = 1e-5
            return 1 / lambda_ * (t - t_tar) ** 2 + lambda_ * (a * c + b * d)

        bounds = [(0, 100), (1, 100), (1, 100)]
        x, _, _ = scipy.optimize.fmin_l_bfgs_b(
            f, [1, 1, 1], bounds=bounds, approx_grad=True)
        b, c, d = x
        return calc_a(b, c, d), b, c, d

    scale_E, scale_I = np.zeros(N), np.zeros(N)
    bias_E, bias_I = np.zeros(N), np.zeros(N)

    for i in range(N):
        scale_E[i], scale_I[i], bias_E[i], bias_I[i] = \
            optimize_single(x_intercept[i], t_spike_tar[i])

    return scale_E, scale_I, bias_E, bias_I


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


def calc_scale_E(scale_I, gL, x_intercept, e_rev_E=4.33, e_rev_I=-0.33):
    """
    For a given scale_I, gL, and x_intercept calculates the corresponding
    scale_E.
    """
    return (scale_I * (e_rev_I * x_intercept - e_rev_I - x_intercept +
                       1) + gL) / (e_rev_E * x_intercept + e_rev_E - x_intercept - 1)


def calc_scale_I(scale_E, gL, x_intercept, e_rev_E=4.33, e_rev_I=-0.33):
    """
    For a given scale_E, gL, and x_intercept calculates the corresponding
    scale_I.
    """
    return (scale_E * (e_rev_E * x_intercept + e_rev_E - x_intercept -
                       1) - gL) / (e_rev_I * x_intercept + e_rev_I - x_intercept + 1)


def max_rate(scale_E, gL, e_rev_E=4.33, e_rev_I=-0.33, tau_ref=2e-3, v_th=1.0):
    """
    Calculates the maximum rate (rate at x=1) of a neuron with conductance based
    neurons for the given scale_E and gL.
    """
    g_tot, e_eq = lif_cond_calc_e_eq_and_g_tot(
        gL, 2 * scale_E, 0, e_rev_E, e_rev_I)
    return _rate(g_tot, e_eq, tau_ref=2e-3, v_th=1.0)


def solve_max_rate_x_intercept(x_intercept, max_rate, gL=None, e_rev_E=4.33,
                               e_rev_I=-0.33, tau_ref=2e-3, v_th=1.0):
    """
    Calculates scale factors for gL, gE, and gI for a conductance based synapse
    tuning curve with maximum rate max_rate and x-intercept x_intercept.

    max_rate: maximum rate of the tuning curve.
    x_intercept: x-intercept of the tuning curve.
    gL: if None, calculates a matching gL. Otherwise, keeps gL constant.
    """

    if gL is None:
        gL, scale_E, scale_I, bias_E, bias_I = calc_gL_scale_E_scale_I(
            x_intercept, max_rate, e_rev_E, e_rev_I, tau_ref), 1, 1
    else:
        scale_E, scale_I, bias_E, bias_I = optimize_scale_E_scale_I_bias_E_bias_I(
            x_intercept, max_rate, e_rev_E, e_rev_I, tau_ref)

    return gL, scale_E, scale_I, bias_E, bias_I
