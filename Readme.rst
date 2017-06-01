Sinha2016
---------

Source code for my Ph.D. simulations.

Requirements
============

- `my fork of NEST <https://github.com/sanjayankur31/nest-simulator>`__.
  It's kept in sync with upstream. The only difference between the two is that my fork does not use the internal implementation of NEST that updates network connectivity. Instead, required algorithms are implemented in the simulation script in Python and connectivity is updated in NEST using the PyNEST API. 
  The NEST development team intends to remove structural plasticity from NEST core in the future. The plan is to develop a standalone structural plasticity manager that will interface with different simulators.

- you can use `my scripts <https://github.com/sanjayankur31/Sinha2016-scripts>`__ to post-process the data
- :code:`python-nose` for a few simple tests

Outputs from the simulation
============================

Spike raster files
~~~~~~~~~~~~~~~~~~~

.. code:: text

    neuronID    spiketime(ms)

- Name format: :code:`spikes-{rank}-{neuron-set}.gdf`
- Collected throughout the simulation
- One file per MPI rank
- These files can be easily combined using the linux sort command to get a combined list of spikes, since each spike raster file is already sorted by time.

For the following neuron sets:

- Excitatory neurons
- Inhibitory neurons
- Pattern/signal neurons
- Background/noise neurons
- Recall neurons
- Deafferentiated neurons
- Stim neurons

Used to generate:

- Global firing rate vs time plots
- ISI CV vs time plots
- Mean firing rate vs time plots
- Population firing rate at particular time plot (snapshots) (WIP)
- SNR value (WIP)
- Raster plots showing E and I spikes

Synaptic weight files
~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    time(ms), comma separated conductances(nS)

    Last line specifies max number of conductance columns

- Name format: :code:`00-synaptic-weights-{synapse-group}-{rank}.txt`
- Collected at particular times - set by the recording period in the simulation
- One file per MPI rank
- The conductances from various rank files will need to be merged. Since these files won't be too large, I can use pandas dataframes to quickly do this.

One file for each of the following synapse groups:

- EE
- EI
- II
- IE

Used to generate:

- Plots with Means and STDs of synaptic conductances
- Also gives an idea of the number of synaptic connections (to compare with synaptic elements to confirm correctness)

Calcium concentration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    time(ms), comma separated calcium values(unit less)

    Last line specifies number of neurons which is the same as the number of calcium values in each line

- Name format: :code:`01-calcium-{neuron-set}-{rank}.txt`
- Collected at particular times - set by the recording period in the simulation
- One file per MPI rank
- Will need to be merged, again using pandas dataframes

For the following neuron sets:

- E neurons
- I neurons

Used to generate:

- Plots showing means + STDs of calcium concentrations


Total synaptic elements files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    time(ms) a_total a_connected d_ex_total d_ex_connected d_in_total d_in_connected

- Name format: :code:`02-synaptic-elements-totals-{neuron-set}-{rank}.txt`
- Collected at particular times - set by the recording period in the simulation
- One file per MPI rank
- Will need to be merged, again using pandas dataframes

For the following neuron sets:

- E neurons
- I neurons

Used to generate:

- Plots showing evolution of various synaptic elements

Per neuron synaptic element files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    neuronID  a_total a_connected d_ex_total d_ex_connected d_in_total d_in_connected

- Name format: :code:`03-synaptic-elements-{neuron-set}-{rank}-{time}.txt`
- Collected at particular times
- New file at each collection time
- One file per MPI rank
- Will need to be merged and sorted - I'll use pandas

For the following neuron sets:

- E neurons
- I neurons

Used to generate:

- Plots showing a snapshot of the network
- Will also come in handy later when we want to look at synaptic elements of particular neurons and particular regions


Per neuron synapse loss files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    time(ms) gid total_conns conns_deleted

- Name format: :code:`04-synapses-deleted-{rank}.txt`
- Collected after synapses are deleted per structural plasticity update
- One file per MPI rank, although all files should be identical

Used to generate:

- Plots showing synapse loss for individual neurons
- Plots showing mean synapse loss for network

Per neuron synapse gain files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    time(ms) gid conns_gained

- Name format: :code:`04-synapses-formed-{rank}.txt`
- Collected after new synapses are formed per structural plasticity update
- One file per MPI rank, although all files should be identical

Used to generate:

- Plots showing synapse gain for individual neurons
- Plots showing mean synapse gain for network

The data from the two together will give:

- Plots showing synaptic turnover as the network evolves
