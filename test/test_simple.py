#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nengo
import numpy as np
import conductance_synapses

with nengo.Network() as model:
    src = nengo.Node(np.sin, label="src")
    ens = nengo.Ensemble(100, 1, label="ens")
    tar = nengo.Node(size_in=1, label="tar")

    nengo.Connection(src, ens)
    nengo.Connection(ens, tar)

    pout = nengo.Probe(tar)

# Run the model a first time
T = 10.0
dt = 1e-4
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)
    data_1 = sim.data[pout]

# Transform the model and run it a second time
with conductance_synapses.transform(
    model, sim=sim,
    use_conductance_synapses=True,
    use_jbias=True,
    use_factorised_weights=False) as model:
    pout = nengo.Probe(tar)

# Run the model a second time
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)
    data_2 = sim.data[pout]

# Print the RMSE
print(np.linalg.norm(data_1 - data_2) / np.sqrt(len(data_1)))

# Plot the results
import matplotlib.pyplot as plt
ts = np.arange(0, T, dt)
fig = plt.figure()
ax = fig.gca()
ax.plot(ts, data_1)
ax.plot(ts, data_2)
ax.set_ylim(-1.5, 1.5)
ax.set_ylabel("Decoded population state")
ax.set_xlabel("Simulation time $t$")
plt.show()
