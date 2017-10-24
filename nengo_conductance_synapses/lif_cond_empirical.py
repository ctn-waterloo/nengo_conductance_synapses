#!/usr/bin/env python3

#   This file is part of soft_cond_lif
#   (c) Andreas Stöckel 2017
#
#   soft_cond_lif is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   soft_cond_lif is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with soft_cond_lif.  If not, see <http://www.gnu.org/licenses/>.

from copy import deepcopy
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from . import sim_cond_exp
import sys, os, json, hashlib, pickle


class LifCondEmpirical:
    """
    The LifCondEmpirical class is responsible for empirically measuring both
    the forward and the inverse tuning curves of a LIF neuron with conductance
    based synapses. This class empirically determines these mappings using a
    multi-threaded C++ neuron simulator backend.
    """

    @staticmethod
    def _show_progress(progress):
        WIDTH = 50
        perc = progress * 100.0
        s = '\r{0:8.1f}'.format((perc)) + '% ['
        for i in range(WIDTH):
            cur = (i * 100 / WIDTH) < perc
            prev = ((i - 1) * 100 / WIDTH) < perc
            if cur and prev:
                s += '='
            elif prev:
                s += '>'
            else:
                s += ' '
        s += ']'
        sys.stdout.write(s)
        sys.stdout.flush()

    def __init__(self,
                 e_rev_e=4.33,
                 e_rev_i=-0.33,
                 w_syn_e=1.0,
                 w_syn_i=1.0,
                 tau_syn_e=5e-3,
                 tau_syn_i=5e-3,
                 tau_refrac=2e-3,
                 gL=50,
                 gE_max=500,
                 gI_max=1500,
                 max_rate=200,
                 min_rate=1,
                 resolution=20,
                 repeat=10,
                 T=10.0,
                 seed=None):

        # Copy some of the parameters
        self.w_syn_e = w_syn_e
        self.w_syn_i = w_syn_i
        self.gE_max = gE_max
        self.gI_max = gI_max
        self.min_rate = min_rate
        self.max_rate = max_rate

        # Use a random seed if None is given as seed
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(np.iinfo(np.int32).max)

        # Translate the parameters to "biological" neuron and synapse parameters
        self.params = sim_cond_exp.translate_simulation_parameters(
            e_rev_e,
            e_rev_i,
            w_syn_e,
            w_syn_i,
            gL,
            params=sim_cond_exp.task(
                tau_refrac=tau_refrac,
                tau_syn_e=tau_syn_e,
                tau_syn_i=tau_syn_i,
                seed=self.seed))
        self.params["input"]["T"] = T
        self.params["input"]["repeat"] = repeat

        # Generate a linear list of gEs, gIs, and rates that should be tested
        self.gEs = np.linspace(0.0, gE_max, resolution)
        self.gIs = np.linspace(0.0, gI_max, resolution)
        self.rates = np.linspace(min_rate, max_rate, resolution)

        # Serialise the task description and hash it
        task_str = json.dumps([self.params, self.w_syn_e, self.w_syn_i, self.gE_max, self.gI_max, self.min_rate, self.max_rate, resolution]).encode('utf-8')
        h = hashlib.sha256(task_str).hexdigest()[0:16]

        fn = '/tmp/.cache_lif_cond_empirical_' + h
        if os.path.isfile(fn):
            # Data is cached, just load it
            with open(fn, 'rb') as f:
                data = pickle.load(f)
            self.forward_map = data['forward_map']
            self.forward_map_stddev = data['forward_map_stddev']
            self.inverse_map_gEs = data['inverse_map_gEs']
            self.inverse_map_gIs = data['inverse_map_gIs']
        else:
            # Calculate the forward mapping for these rates
            self.forward_map, self.forward_map_stddev = \
                self._calculate_forward_mapping()

            # Calculate the inverse map
            self.inverse_map_gEs, self.inverse_map_gIs = \
                self._calculate_inverse_mapping()

            # Cache the result
            with open(fn, 'wb') as f:
                pickle.dump({
                    'forward_map': self.forward_map,
                    'forward_map_stddev': self.forward_map_stddev,
                    'inverse_map_gEs': self.inverse_map_gEs,
                    'inverse_map_gIs': self.inverse_map_gIs
                }, f)

        # Setup the interpolators
        self.foward_map_ipol = \
            RegularGridInterpolator((self.gEs, self.gIs), self.forward_map.T, bounds_error=False, fill_value=None)
        self.forward_map_stddev_ipol = \
            RegularGridInterpolator((self.gEs, self.gIs), self.forward_map_stddev.T, bounds_error=False, fill_value=None)
        self.inverse_map_gEs_ipol = \
            RegularGridInterpolator((self.gIs, self.rates), self.inverse_map_gEs, bounds_error=False, fill_value=None)
        self.inverse_map_gIs_ipol = \
            RegularGridInterpolator((self.gEs, self.rates), self.inverse_map_gIs, bounds_error=False, fill_value=None)

    def rate(self, gE, gI):
        return self.foward_map_ipol((gE, gI)).clip(0, None)

    def calc_gE_for_rate(self, rate, gI):
        return self.inverse_map_gEs_ipol((rate, gI)).clip(0, None)

    def calc_gI_for_rate(self, rate, gE):
        return self.inverse_map_gIs_ipol((rate, gE)).clip(0, None)

    def _calculate_forward_mapping(self):
        # Calculate a 2D grid of gEs and gIs
        mesh_gEs, mesh_gIs = np.meshgrid(self.gEs, self.gIs)
        mesh_shape = mesh_gEs.shape
        N = np.prod(mesh_shape)

        mesh_gEs, mesh_gIs = mesh_gEs.flatten(), mesh_gIs.flatten()
        mesh_rate = np.zeros(mesh_shape).flatten()
        mesh_stddev = np.zeros(mesh_shape).flatten()

        # Submit all experiments to the backend
        counter = {'count': 0}
        with sim_cond_exp.SimCondExp() as sim:
            for i in range(0, N):
                # Setup the the simulation
                task = deepcopy(self.params)
                task["input"]["rate_e"] = mesh_gEs[i] / self.w_syn_e
                task["input"]["rate_i"] = mesh_gIs[i] / self.w_syn_i

                # Submit the simulation to the simulator, on completion write the result to
                # the mesh_rate array and update the progress bar
                def make_handle_response(i):
                    def handle_response(task, result):
                        # Store the result
                        mesh_rate[i] = np.mean(result)
                        mesh_stddev[i] = np.sqrt(np.var(result))

                        # Display progress
                        counter["count"] += 1
                        self._show_progress(counter["count"] / N)

                    return handle_response

                sim.submit(task, make_handle_response(i))

        sys.stdout.write('\n')
        sys.stdout.flush()

        return (mesh_rate.reshape(mesh_shape), mesh_stddev.reshape(mesh_shape))

    def _calculate_inverse_mapping(self):
        # Maximum number of iterations
        MAX_IT = 20

        # Maximum delta between upper and lower bound before finishing
        MAX_DELTA = 1

        # Calculate two 2D grids for gE and rate and gI and rate
        mesh_gEs, mesh_rates = np.meshgrid(self.gEs, self.rates)
        mesh_gIs, _ = np.meshgrid(self.gIs, self.rates)
        mesh_shape = mesh_gEs.shape
        N = np.prod(mesh_shape)

        mesh_gEs = mesh_gEs.flatten()
        mesh_gIs = mesh_gIs.flatten()
        mesh_rates = mesh_rates.flatten()

        mesh_res_gE = np.empty(N)
        mesh_res_gI = np.empty(N)
        mesh_res_gE[:] = np.nan
        mesh_res_gI[:] = np.nan

        # Initialize the pool collecting all empirical runs
        forward_map = self.forward_map.flatten()
        forward_map_gEs, forward_map_gIs = np.meshgrid(self.gEs, self.gIs)
        forward_map_gEs = forward_map_gEs.flatten()
        forward_map_gIs = forward_map_gIs.flatten()
        max_items = (2 * MAX_IT + 1) * N
        pool = np.zeros((max_items, 3))
        pool[0:N, :] = np.array((forward_map_gEs, forward_map_gIs, forward_map)).T

        counter = {'count': 0, 'tasks_done': 0, 'idx': N}

        # Function finding upper and lower bounds for gE/gI given a rate
        def bounds(solve_for, gE, gI, rate):
            p = pool[0:counter['idx']] # Slice the valid subset of "pool"

            def do_find_bounds(i, m1, m2, g):
                q = p[np.logical_and(m1 * p[:, i] >= m1 * g, m2 * rate >=
                                     m2 * p[:, 2]), 1 - i]
                if q.size == 0:
                    return None
                return q[np.argmin(m1 * q)]

            if solve_for == "gE":
                lower = do_find_bounds(1, -1, 1, gI)
                upper = do_find_bounds(1, 1, -1, gI)
            elif solve_for == "gI":
                lower = do_find_bounds(0, -1, -1, gE)
                upper = do_find_bounds(0, 1, 1, gE)
            return lower, upper

        # Submit the tasks to the backend
        self._show_progress(0)
        max_tasks = 2 * MAX_IT * N
        with sim_cond_exp.SimCondExp() as sim:
            for i in range(0, N):

                def make_handle_response(i, it, solve_for, cur_max, gE, gI,
                                         rate):
                    def handle_response(task, result):
                        # Append the result to the result collection
                        if not result is None:
                            pool[counter["idx"], :] = (gE, gI, np.mean(result))
                            counter["tasks_done"] += 1
                            counter["idx"] += 1

                        # Get an upper and lower bound for the variable we're
                        # solving for
                        lower, upper = bounds(solve_for, gE, gI, rate)
                        if lower is None:
                            lower = 0.0
                        if upper is None:
                            upper = cur_max * 1.5
                        tar = 0.5 * (upper + lower)

                        # Abort for large numbers of iterations or if
                        # upper/lower are closing in on the
                        low_err = (upper - lower) < MAX_DELTA
                        if it >= MAX_IT or low_err:
                            if low_err:
                                if solve_for == "gE":
                                    mesh_res_gE[i] = tar
                                elif solve_for == "gI":
                                    mesh_res_gI[i] = tar
                            counter["count"] += 1
                            counter["tasks_done"] += (MAX_IT - it)
                            self._show_progress(counter["tasks_done"] / max_tasks)
                            return

                        # Setup the the simulation
                        task = deepcopy(self.params)
                        if solve_for == "gE":
                            task["input"]["rate_e"] = tar / self.w_syn_e
                            task["input"]["rate_i"] = gI / self.w_syn_i
                            sim.submit(task,
                                       make_handle_response(
                                           i, it + 1, solve_for,
                                           max(cur_max, upper), tar, gI, rate))
                        elif solve_for == "gI":
                            task["input"]["rate_e"] = gE / self.w_syn_e
                            task["input"]["rate_i"] = tar / self.w_syn_i
                            sim.submit(task,
                                       make_handle_response(
                                           i, it + 1, solve_for,
                                           max(cur_max, upper), gE, tar, rate))

                        self._show_progress(counter["tasks_done"] / max_tasks)


                    return handle_response

                # Kickstart the process by issuing dummy responses
                make_handle_response(i, 0, "gI", self.gI_max, mesh_gEs[i],
                                     None, mesh_rates[i])(None, None)
                make_handle_response(i, 0, "gE", self.gE_max, None,
                                     mesh_gIs[i], mesh_rates[i])(None, None)

            # Process all requests, wait until all grid points have been
            # processed
            while sim.wait():
                if counter["count"] == 2 * N:
                    break

        sys.stdout.write('\n')
        sys.stdout.flush()

        return (mesh_res_gE.reshape(mesh_shape),
                mesh_res_gI.reshape(mesh_shape))

