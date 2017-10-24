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
import json
import hashlib
import os
import subprocess
import numpy as np
import shelve
import threading
import time
import pexpect


def task(cm=1e-9,
         g_leak=5e-8,
         v_thresh=-50e-3,
         v_rest=-65e-3,
         v_reset=-65e-3,
         tau_refrac=2e-3,
         tau_syn_e=5e-3,
         w_syn_e=0.1e-6,
         e_rev_syn_e=0e-3,
         tau_syn_i=5e-3,
         w_syn_i=0.1e-6,
         e_rev_syn_i=-75e-3,
         rate_e=0.0,
         rate_i=0.0,
         repeat=10,
         spike_loss_e=0.0,
         spike_loss_i=0.0,
         T=10.0,
         seed=None):
    """
    Assembles a Task data structure that will be sent to the neuron simulator
    process. The given id is used to associate individual experiment results
    with the original task.
    """

    # Use a random seed if None is given as seed
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)

    return {
        "neuron": {
            "cm": cm,
            "g_leak": g_leak,
            "v_thresh": v_thresh,
            "v_rest": v_rest,
            "v_reset": v_reset,
            "tau_refrac": tau_refrac
        },
        "syn_e": {
            "tau_syn": tau_syn_e,
            "w_syn": w_syn_e,
            "e_rev_syn": e_rev_syn_e
        },
        "syn_i": {
            "tau_syn": tau_syn_i,
            "w_syn": w_syn_i,
            "e_rev_syn": e_rev_syn_i
        },
        "input": {
            "id": "",
            "T": T,
            "rate_e": rate_e,
            "rate_i": rate_i,
            "repeat": repeat,
            "spike_loss_e": spike_loss_e,
            "spike_loss_i": spike_loss_i,
            "seed": seed
        }
    }


def translate_simulation_parameters(e_rev_e=4.33,
                                    e_rev_i=-0.33,
                                    w_syn_e=1.0,
                                    w_syn_i=1.0,
                                    gL=50,
                                    tau_syn_e=None,
                                    tau_syn_i=None,
                                    params=task()):
    params = deepcopy(params)

    vTh = params["neuron"]["v_thresh"]
    eL = params["neuron"]["v_rest"]
    Cm = params["neuron"]["cm"]
    tauE = params["syn_e"]["tau_syn"] if tau_syn_e is None else tau_syn_e
    tauI = params["syn_i"]["tau_syn"] if tau_syn_i is None else tau_syn_i

    params["neuron"]["g_leak"] = gL * Cm
    params["syn_e"]["w_syn"] = w_syn_e * Cm / tauE
    params["syn_e"]["e_rev_e"] = (vTh - eL) * e_rev_e + eL
    params["syn_i"]["w_syn"] = w_syn_i * Cm / tauI
    params["syn_i"]["e_rev_i"] = (vTh - eL) * e_rev_i + eL

    return params


class SimCondExp:
    def __init__(self):
        # Make sure the simulator is compiled
        curdir = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(["make", "-s", "-C", curdir,
                               "all"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            stdout, stderr = process.communicate()
            if process.wait() != 0:
                raise Exception(
                    "Error while compiling C++ simulator backend:\n" + stderr.decode('utf8')
                )

        # Initialize the persistent cache
        try:
            self.results = shelve.open("/tmp/.cache_sim_cond_exp_pipe.db")
        except:
            self.results = {}

        # Initialize internal caches
        self.tasks = {}

        # Start the subprocess
        executable = '/tmp/sim_cond_exp_pipe'
        self.process =  subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.open = True

        # Spawn a thread for writing to the process
        self.write_queue = []
        def write_thread_proc():
            while True:
                while len(self.write_queue) > 0:
                    self.process.stdin.write(self.write_queue.pop(0))
                    self.process.stdin.flush()
                if not self.open:
                    break
                time.sleep(10e-3)
        self.write_thread = threading.Thread(target=write_thread_proc)
        self.write_thread.start()

    def submit(self, task, callback):
        # Make sure the simulator subprocess is still alive
        if not self.open:
            raise Exception("Simulator instance is closed")

        # Do not modify the task
        task = deepcopy(task)

        # Discretize rate_e and rate_i to two decimal places, increasing the
        # number of cache hits
        task["input"]["rate_e"] = int(100 * task["input"]["rate_e"]) / 100
        task["input"]["rate_i"] = int(100 * task["input"]["rate_i"]) / 100

        # Serialise the task description and hash it
        task_str = json.dumps(task).encode('utf-8')
        h = hashlib.sha256(task_str).hexdigest()[0:16]

        # If the task was already executed just call the callback and return
        if h in self.results:
            callback(task, self.results[h])
            return

        # If a callback is already registered for this task, append it to the
        # list of callbacks
        if h in self.tasks:
            self.tasks[h]["callbacks"].append(callback)
            return

        # Register the callback for the given hash
        self.tasks[h] = {"task": task, "callbacks": [callback]}

        # Submit the task to the subprocess
        task["input"]["id"] = h
        task_str = json.dumps(task).encode('utf-8')
        self.write_queue.append(task_str + b'\n')

    def wait(self):
        if self.open:
            response = self.process.stdout.readline()
            if response == "":
                # Wait for the write thread to finish
                self.open = False
                self.write_thread.join()

                # Explicitly close stdin
                self.process.stdin.close()

                self.process.wait()
                return False
            self.handle_response(response)
            return True
        return False

    def handle_response(self, response):
        # Split the response at the comma
        if (len(response) == 0):
            return
        data = response.decode('utf8')[:-1].split(',')

        # Extract the task id
        h, result = data[0], np.array(list(map(float, data[1:])))
        if h in self.tasks:
            # Call all associated callbacks
            task = self.tasks[h]
            for callback in task["callbacks"]:
                callback(task["task"], result)

            # Store the result for future reference
            self.results[h] = result

            del self.tasks[h]

    def close(self):
        if self.open:
            # Mark this instance as closed
            self.open = False
            self.write_thread.join()

            # Explicitly close stdin
            self.process.stdin.close()

            # Wait for final responses and handle them
            for response in self.process.stdout:
                self.handle_response(response)

            # Wait for the process to exit and explicitly close it
            self.process.wait()

            # Close the database
            if hasattr(self.results, 'close'):
                self.results.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

