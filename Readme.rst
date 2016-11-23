Sinha2016
---------

My Ph.D. simulations


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
