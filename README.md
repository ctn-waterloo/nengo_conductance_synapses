# Fun with conductance based synapses in Nengo

`conductance_synapses.py` provides a fairly generic function `transform` which
transforms a Nengo model into a Nengo model in which all LIF ensembles are
converted to LIF ensembles with inhibitory and excitatory conductance based
synapses.

See the `test` folder for a bunch of tests and usage examples
(work in progress).

