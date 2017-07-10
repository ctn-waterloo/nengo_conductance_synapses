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

    # Mask out invalid values
    mask = 1 * np.logical_or(
        np.logical_and(e_eq > 1, g_tot > 0),
        np.logical_and(e_eq < 0, g_tot < 0))
    t_spike = -np.log1p(-mask * v_th / (mask * e_eq + (1 - mask))) / g_tot
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
    alpha = np.log((e_rev_E - 1) / e_rev_E)
    return (alpha * e_rev_E + 1) / alpha


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

    rate: target rate. Either a scalar or an arbitrary matrix. If both rate and
    gI are a matrix, rate and gI must have the same shape.
    gL: leak conductance, must be a scalar.
    gI: inhibitory conductance. May be a scalar or a matrix. If both gI and rate
    are a matrix, they must both have the same shape and a vector of that size
    is returned.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    atol: absolute tolerance.
    """

    # Convert rate/gI to matrices, make sure they have equal size
    rates = np.atleast_1d(rate).astype(np.float)
    gIs = np.atleast_1d(gI).astype(np.float)
    assert (rates.size == 1) or (gIs.size == 1) or (rates.size == gIs.size), \
        "One of rate, gI must be scalar or rate and gI must have the same size"
    if rates.size > 1 and gIs.size == 1:
        gIs = np.ones_like(rates) * gIs
    elif rates.size == 1 and gIs.size > 1:
        rates = np.ones_like(gIs) * rates

    # Shorthands
    EE = e_rev_E
    EI = e_rev_I

    # Solve t_spike_cond for gI using Newton's method
    xs = np.ones_like(gIs) * 1000.0
    for gI, rate, x in np.nditer((gIs, rates, xs), [],
                                 [['readonly'], ['readonly'], ['readwrite']]):
        # Convert rate to time-to-spike and special case handling
        if rate > 1 / tau_ref:
            s = 1e-6  # assume quite small time-to-spike
        if rate <= 0:
            x[...] = 0
            continue
        else:
            s = 1 / rate - tau_ref  # rate to time-to-spike

        step = 1
        while np.abs(step) > atol:
            e = np.exp(-s * (gL + gI + x))
            f = (x * EE + gI * EI) * e - x * \
                EE - gI * EI + x + gI + gL
            df = (-x * EE * s - gI * EI * s + EE) * e - EE + 1

            step = f / df
            x[...] = x - step
    return xs


def calc_gI_for_rate(rate,
                     gL,
                     gE,
                     e_rev_E=4.33,
                     e_rev_I=-0.33,
                     tau_ref=2e-3,
                     atol=1e-9):
    """
    Calculates the inhibitory conductivity required to achive a certain spike
    rate given an inhibitory conductivity.

    gL: leak conductance.
    gE: excitatory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    atol: absolute tolerance.
    """

    # Just swap e_rev_E and e_rev_I and call the previous function.
    return calc_gE_for_rate(rate, gL, gE, e_rev_I, e_rev_E, tau_ref, atol)


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
        gEoffsNew = calc_gEoffs(gE, gI) * 0.1 + gEoffs * 0.9

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


def get_activities(eval_points,
                   encoders,
                   a,
                   b,
                   c,
                   d,
                   gL,
                   tau_ref=2e-3,
                   e_rev_E=4.33,
                   e_rev_I=-0.33):
    """
    Calculates the activites of a neuron population for the given evaluation
    points. Parameters are the eval_points, the neuron encoders, as well as the
    population parameters a, b, c, d, tau_rc, tau_ref.

    eval_points: matrix containing X vectors for which the activities should be
    evaluated.
    encoders: encoder matrix of this neuron population.
    a: excitatory conductance gain.
    b: excitatory conductance offset.
    c: inhibitory conductance gain.
    d: inhibitory conductance offset.
    tau_rc: membrane time constant.
    tau_ref: refractory period.
    e_rev_E: excitatory reversal potential.
    e_rev_I: inhibitory reversal potential.
    """
    X = eval_points @ encoders
    gE = np.maximum(0, a[None, :] * X + b[None, :])
    gI = np.maximum(0, c[None, :] * X + d[None, :])

    return lif_cond_rate(gL, gE, gI, e_rev_E, e_rev_I, tau_ref)


def solve_weight_matrices_for_activities(pre_activities,
                                         post_activities,
                                         gL=50,
                                         e_rev_E=4.33,
                                         e_rev_I=-0.33,
                                         tau_ref=2e-3,
                                         reg=1e-1,
                                         atol=1e-3):
    """
    Calculates non-negative excitatory and inhibitory weight matrices such that
    the pre and post activities line up.
    """

    import scipy.optimize

    m = pre_activities.shape[0]  # Number of samples
    Npre = pre_activities.shape[1]  # Number of neurons in the pre-population
    Npost = post_activities.shape[
        1]  # Number of neurons in the post-population
    assert pre_activities.shape[0] == post_activities.shape[0], \
        "Number of samples must be equal in pre- and post-activities"

    # Calculate the scaling factor used to translate negative excitator weights
    # to the approximate inhibitory weights
    EV = average_membrane_potential_estimate(e_rev_E)
    scale_I = 1 / (-e_rev_I + EV)

    # Aliases
    Apre = pre_activities
    Apost = post_activities

    # Create the excitatory and inhibitory weight matrix
    WE = np.zeros((Npre, Npost))
    WI = np.zeros((Npre, Npost))
    gE = np.zeros((m, Npost))
    gI = np.zeros((m, Npost))

#    # Calculate the minimum gE needed for output spikes
#    min_gE = calc_gE_for_intercept(gL, 0, e_rev_E=4.33, e_rev_I=-0.33)

    # Form the Gram matrix and apply regularization.
    sigma = (np.max(Apre) * reg) ** 2
    GApre = Apre.T @ Apre
    np.fill_diagonal(GApre, GApre.diagonal() + m * sigma)

    def solve_for_weights(gTar, non_negative=False):
        W = np.zeros((Npre, Npost))
        GgTar = Apre.T @ gTar
        for i in range(Npost):
            if non_negative:
                W[:, i] = scipy.optimize.nnls(GApre, GgTar[:, i])[0]
            else:
                W[:, i] = scipy.optimize.lsq_linear(GApre, GgTar[:, i]).x
        return W

    # Start with an initial guess for WE and WI
    WE = solve_for_weights(calc_gE_for_rate(
        Apost,
        gL=gL,
        gI=gI,
        e_rev_E=e_rev_E,
        e_rev_I=e_rev_I,
        tau_ref=tau_ref,
        atol=atol))
    neg_idx = WE < 0
    WI[neg_idx] = -WE[neg_idx] * scale_I
    WE[neg_idx] = 0

#    gI = Apre @ WI

#    for _ in range(4):
#        # Refine gE and gI using NNLS
#        WE = solve_for_weights(calc_gE_for_rate(
#            Apost,
#            gL=gL,
#            gI=gI,
#            e_rev_E=e_rev_E,
#            e_rev_I=e_rev_I,
#            tau_ref=tau_ref,
#            atol=atol), True)
#        gE = Apre @ WE

#        # Refine gE and gI using NNLS
#        WI = solve_for_weights(calc_gI_for_rate(
#            Apost,
#            gL=gL,
#            gE=gE,
#            e_rev_E=e_rev_E,
#            e_rev_I=e_rev_I,
#            tau_ref=tau_ref,
#            atol=atol), True)
#        gI = Apre @ WI


    # Calculate the error
    err = np.sqrt(
        np.mean((Apost - lif_cond_rate(
            gL, Apre @ WE, Apre @ WI, e_rev_E, e_rev_I, tau_ref))**2))

    print(err)

    return WE, WI


def solve_weight_matrices(pre_activities,
                          targets,
                          encoders,
                          a,
                          b,
                          c,
                          d,
                          reg=0.1,
                          non_negative=True):
    """
    Calculates the excitatory and inhibitory weight matrices such that the post
    population of a connection represents the target value given whenever the
    pre-population has the given activities.
    """

    import scipy.optimize

    m = pre_activities.shape[0]  # Number of samples
    Npre = pre_activities.shape[1]  # Number of neurons in the pre-population
    Npost = encoders.shape[1]  # Number of neurons in the post-population
    D = encoders.shape[0]  # Dimensions represented in the post-population

    a, b, c, d = (np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(c),
                  np.atleast_1d(d))
    assert a.ndim == b.ndim == c.ndim == d.ndim == 1, \
        "Parameter vectors a, b, c, d must be one-dimensional"
    assert a.shape[0] == b.shape[0] == c.shape[0] == d.shape[0] == Npost, \
        "Length of the parameter vectors a, b, c, d does not match the " + \
        "number of neurons in the post-population"
    assert D == targets.shape[1], \
        "Encoder and target dimensionality do not match"
    assert pre_activities.shape[0] == targets.shape[0], \
        "Samples must be equal for pre_activities and targets"

    # Calculate the actual target functions for gE and gI. Clip to zero.
    # The decoded function can never be smaller than zero.
    YE = targets @ encoders
    gE_target = np.maximum(0, a[None, :] * YE + b[None, :])
    gI_target = np.maximum(0, c[None, :] * YE + d[None, :])

    # Fetch the pre-population activities
    A = pre_activities

    # Form the Gram matrix and apply regularization.
    # Code adapted from nengo/solvers.py, NnlsL2
    sigma = np.max(A) * reg
    GA = A.T @ A
    np.fill_diagonal(GA, GA.diagonal() + m * sigma**2)
    GgE = A.T @ gE_target
    GgI = A.T @ gI_target

    WE, WI = np.zeros((2, Npre, Npost))
    for i in range(Npost):
        if non_negative:
            WE[:, i] = scipy.optimize.nnls(GA, GgE[:, i])[0]
            WI[:, i] = scipy.optimize.nnls(GA, GgI[:, i])[0]
        else:
            WE[:, i] = scipy.optimize.lsq_linear(GA, GgE[:, i]).x
            WI[:, i] = scipy.optimize.lsq_linear(GA, GgI[:, i]).x

    return WE, WI

