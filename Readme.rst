Sinha2016
---------

Source code for my Ph.D. simulations.

Requirements
============

- `my fork of NEST <https://github.com/sanjayankur31/nest-simulator>`__.
  It's kept in sync with upstream. The only difference between the two is that my fork does not use the internal implementation of NEST that updates network connectivity. Instead, required algorithms are implemented in the simulation script in Python and connectivity is updated in NEST using the PyNEST API. 
  The NEST development team intends to remove structural plasticity from NEST core in the future. The plan is to develop a standalone structural plasticity manager that will interface with different simulators.

- you can use `my scripts <https://github.com/sanjayankur31/Sinha2016-scripts>`__ to post-process the data

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

- LPZ centre E
- LPZ border E
- Peri LPZ E
- LPZ centre I
- LPZ border I
- Peri LPZ I
- Pattern/signal neurons
- Background/noise neurons
- Recall neurons
- Stim neurons

Used to generate:

- Mean firing rate vs time plots
- ISI CV vs time plots
- Population firing rate at particular time plot (snapshots)
- SNR value
- Raster plots showing E and I spikes

Calcium concentration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    neuron_ID Ca_concentration

    tab seprated

- Name format: :code:`02-calcium-{neuron-set}-{rank}-{time}.txt`
- One file per region per recording time per rank.
- Will need to be merged, again using pandas dataframes

For the following neuron sets:

- LPZ centre E
- LPZ border E
- Peri LPZ E
- LPZ centre I
- LPZ border I
- Peri LPZ I

Used to generate:

- Plots showing means + STDs of calcium concentrations for different regions.


Per neuron synaptic element files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    neuronID  a_total a_connected d_ex_total d_ex_connected d_in_total d_in_connected

- Name format: :code:`03-synaptic-elements-{neuron-set}-{rank}-{time}.txt`
- Collected at particular times
- New file at each collection time
- One file per MPI rank

For the following neuron sets:

- LPZ centre E
- LPZ border E
- Peri LPZ E
- LPZ centre I
- LPZ border I
- Peri LPZ I

Used to generate:

- Plots showing a snapshot of the network
- Evolution of synaptic elements for different regions at different times.

Per neuron synapse loss files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    gid total_conns conns_deleted

- Name format: :code:`04-synapses-deleted-{region}-{rank}.txt`
- Collected after synapses are deleted per structural plasticity update
- One file per region
- Only published by rank 0 (since all ranks would publish identical data)


For each of these regions:

- LPZ centre E
- LPZ border E
- Peri LPZ E
- LPZ centre I
- LPZ border I
- Peri LPZ I


Used to generate:

- Plots showing synapse loss for individual neurons
- Plots showing mean synapse loss for network

Per neuron synapse gain files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    gid conns_gained

- Name format: :code:`04-synapses-formed-{region}-{rank}.txt`
- Collected after new synapses are formed per structural plasticity update
- One file per MPI rank, although all files should be identical

For each of these regions:

- LPZ centre E
- LPZ border E
- Peri LPZ E
- LPZ centre I
- LPZ border I
- Peri LPZ I


Used to generate:

- Plots showing synapse gain for individual neurons
- Plots showing mean synapse gain for network

The data from the two together will give:

- Plots showing synaptic turnover as the network evolves


Network connection information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    src target weight

- Name format: :code:`08-syn_conns-{synapse type}-{rank}-{simtime}.txt`
- Collected at regular intervals

For each synapse type:

- EE
- EI
- IE
- II


Used to generate:

- Plots showing conductances input to each region, mean and total
- Plots showing incoming synapse numbers to neurons in different regions
