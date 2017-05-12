# Fun with conductance based synapses in Nengo

The `nengo_conductance_synapses` package provides a fairly generic function
`transform` which transforms a Nengo model into a Nengo model in which all LIF
ensembles are converted to LIF ensembles with inhibitory and excitatory
conductance based synapses.

## How to use

Create and populate a Nengo `Network` instance which you want to transform. Then
simply call

```python
import nengo_conductance_synapses as conductance_synapses
net_out = conductance_synapses.transform(net_in, dt)
```

where `net_in` is the network you want to transform, `net_out` is the target
network, and `dt` is the timstep used for the internal neuron simulation. When
simulating the network, exactly the same timestep as specified here must be
passed to the simulator. A recommended value is `dt = 1e-4`.

Other options that can be passed to the `transform` function include

* `e_rev_E` Excitatory synapse reversal potential (default: 4.33, eqiv. 0mV)
* `e_rev_I` Inhibitory synapse reversal potential (default: -0.33, equiv. -70mV)
* `use_linear_avg_pot` Use a simplified linear approximation to the average membrane potential (default: False)
* `use_conductance_synapses` If set to false, uses normal current based synapses. Network transformation should not change the result (if `use_factorised_weights` and `use_jbias` are set to False as well). This is useful for testing. (default: True)
* `use_factorised_weights` Factorises the internal weight matrix in order to speed up the simulation. (default: False)
* `use_jbias` If false, decodes the bias current from the pre-population of each ensemble, except for those which receive input from nodes only. (default: False)
* `seed` Random seed to be used for the transformation.

Note that all membrane potentials are normalised to a range from 0 to 1, where 0
is the resting and reset potential and 1 is the threshold potential.

### Preventing transformation

You can manually set the attribute `use_conductance_synapses` to `False` on an
ensemble in order to suppress the conversion of this particular LIF ensemble to
an LIF ensemble with conductance based synapses.

```python
with nengo.Network() as net_in:
	ens = nengo.Ensemble(N, D)
	set_attr(ens, 'use_conductance_synapses', False)

net_out = conductance_synapses.transform(net_in, dt)
```

### Unsupported Nengo features

The following Nengo features are currently not supported by the script

* *Semantic Pointer Architecture networks* (SPA). Theoretically this /should/ work, but for now there are some mysterious crashes.
* *Modulatory Learning Rule connections*. This will be quite hard to implement since the script operates on weight matrices and does not preserve the individual decoding vectors which are e.g. modified by the PES rule.

