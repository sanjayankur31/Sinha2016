#!/usr/bin/env python
"""
NEST implementation of Vogels et al. model.

File: Sinha2016.py

Copyright 2015 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import nest
from matplotlib import pyplot as plt
import numpy
import math
import random


class Sinha2016:

    """Simulations for my PhD 2016."""

    def __init__(self):
        """Initialise variables."""
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1
        # start with a smaller population
        self.populations = {'E': 8000, 'I': 2000, 'P': 800, 'R': 200,
                            'EXT': 1000}
        self.numpats = 1
        # Global sparsity
        self.sparsity = 0.02
        self.sparsityExtE = 0.05
        # Calculate connectivity - must be an integer
        self.connectionNumberEE = int((self.populations['E']**2) *
                                      self.sparsity)
        self.connectionNumberII = int((self.populations['I']**2) *
                                      self.sparsity)
        self.connectionNumberIE = int((self.populations['I'] *
                                       self.populations['E']) * self.sparsity)
        self.connectionNumberEI = self.connectionNumberIE
        self.connectionNumberExtE = int((self.populations['EXT'] *
                                         self.populations['R'])
                                        * self.sparsityExtE)

        self.connDictEE = {"rule": "fixed_total_number",
                           "N": self.connectionNumberEE}
        self.connDictEI = {"rule": "fixed_total_number",
                           "N": self.connectionNumberEI}
        self.connDictII = {"rule": "fixed_total_number",
                           "N": self.connectionNumberII}
        self.connDictIE = {"rule": "fixed_total_number",
                           "N": self.connectionNumberIE}
        self.connDictExtE = {"rule": "fixed_total_number",
                             "N": self.connectionNumberExtE}

        # Documentation says things are normalised in the iaf neuron so that
        # weight of 1 translates to 1nS
        self.synDictE = {"weight": 3.}
        self.synDictII = {"weight": -30.}

        self.synDictIE = {"weight": -7., "Wmax": -30000.,
                          'alpha': .32, 'eta': -0.001,
                          'tau': 20.}

        # see the aif source for symbol definitions
        self.neuronDict = {'I_e': 220.0, 'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.}

        self.rank = nest.Rank()

        # Get the number of spikes in these files and then post-process them to
        # get the firing rate and so on
        self.spike_detector_paramsE = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-E'
        }
        self.spike_detector_paramsP = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-pattern'
        }
        self.spike_detector_paramsR = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-recall'
        }
        self.spike_detector_paramsI = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-I'
        }

        self.synaptic_weights_file_name = ("00-synaptic-weights-" +
                                           str(self.rank) + ".txt")

        self.patterns = []
        self.sdP = []
        self.sdR = []
        self.pattern_spike_count_file_names = []
        self.pattern_spike_count_files = []
        self.pattern_count = 0

        self.synaptic_weights_file = open(self.synaptic_weights_file_name, 'w')

        random.seed(42)

    def setup_simulation(self):
        """Set up simulation."""
        # Nest stuff
        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_ERROR')
        # unless using the cluster, just use 24 local threads
        # still gives out different spike files because they're different
        # virtual processes
        # Using 1 thread per core, and 24 MPI processes because I want 24
        # different firing rate files - if I don't use MPI, I only get one
        # firing rate file and I'm not sure how the 24 processes each will
        # write to it
        nest.SetKernelStatus(
            {
                'resolution': self.dt,
                'local_num_threads': 1
            }
        )
        # Set up TIF neurons
        # Setting up two models because then it makes it easier for me to get
        # them when I need to set up patterns
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults("tif_neuronE", self.neuronDict)
        nest.CopyModel("iaf_cond_exp", "tif_neuronI")
        nest.SetDefaults("tif_neuronI", self.neuronDict)

        self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
        self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        # set up synapses
        nest.CopyModel("vogels_sprekeler_synapse", "inhibitory_plastic",
                       self.synDictIE)
        nest.CopyModel("static_synapse", "inhibitory_static",
                       self.synDictII)
        nest.CopyModel("static_synapse", "excitatory_static",
                       self.synDictE)

        nest.Connect(self.neuronsE, self.neuronsE, conn_spec=self.connDictEE,
                     syn_spec="excitatory_static")
        nest.Connect(self.neuronsE, self.neuronsI, conn_spec=self.connDictEI,
                     syn_spec="excitatory_static")
        nest.Connect(self.neuronsI, self.neuronsI, conn_spec=self.connDictII,
                     syn_spec="inhibitory_static")
        nest.Connect(self.neuronsI, self.neuronsE, conn_spec=self.connDictIE,
                     syn_spec="inhibitory_plastic")

        self.sdE = nest.Create('spike_detector',
                               params=self.spike_detector_paramsE)
        self.sdI = nest.Create('spike_detector',
                               params=self.spike_detector_paramsI)

        nest.Connect(self.neuronsE, self.sdE)
        nest.Connect(self.neuronsI, self.sdI)

    def run_simulation(self, simtime, step=False):
        """Run the simulation."""
        sim_steps = numpy.arange(0, simtime)
        if step:
            print("Stepping through the simulation one second at a time")
            for i, step in enumerate(sim_steps):

                nest.Simulate(1000)

                conns = nest.GetConnections(target=self.neuronsE,
                                            source=self.neuronsI)
                weightsIE = nest.GetStatus(conns, "weight")
                mean_weightsIE = numpy.mean(weightsIE)

                conns = nest.GetConnections(target=self.neuronsI,
                                            source=self.neuronsI)
                weightsII = nest.GetStatus(conns, "weight")
                mean_weightsII = numpy.mean(weightsII)

                conns = nest.GetConnections(target=self.neuronsI,
                                            source=self.neuronsE)
                weightsEI = nest.GetStatus(conns, "weight")
                mean_weightsEI = numpy.mean(weightsEI)

                conns = nest.GetConnections(target=self.neuronsE,
                                            source=self.neuronsE)
                weightsEE = nest.GetStatus(conns, "weight")
                mean_weightsEE = numpy.mean(weightsEE)

                statement_w = "{0}\t{1}\t{2}\t{3}\n".format(mean_weightsEE,
                                                            mean_weightsEI,
                                                            mean_weightsII,
                                                            mean_weightsIE)

                self.synaptic_weights_file.write(statement_w)
                self.synaptic_weights_file.flush()
        else:
            print("Not stepping through it one second at a time")
            nest.Simulate(simtime*1000)

            print("Simulation time: "
                  "{} ms".format(nest.GetKernelStatus()['time']))

    def store_pattern(self):
        """ Store a pattern and set up spike detectors."""
        local_neurons = nest.GetNodes(
            nest.CurrentSubnet(), {'model': 'tif_neuronE'},
            local_only=False)[0]

        pattern_neurons = random.sample(
            local_neurons,
            self.populations['P'])

        # strengthen connections
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        nest.SetStatus(connections, {"weight": 24.})

        # store these neurons
        self.patterns.append(pattern_neurons)

        # set up spike detector
        pattern_spike_detector = nest.Create(
            'spike_detector', params=self.spike_detector_paramsP)
        nest.Connect(pattern_neurons, pattern_spike_detector)
        # save the detector
        self.sdP.append(pattern_spike_detector)

        # set up files
        self.pattern_count += 1
        print("Number of patterns stored: {}".format(self.pattern_count))

    def setup_pattern_for_recall(self, pattern_number):
        """
        Set up a pattern for recall.

        All I'm doing is increasing the bg_current for the recall subset.
        """
        recall_neurons = self.patterns[pattern_number - 1]
        nest.SetStatus(recall_neurons, {'I_e': 500.0})

        recall_spike_detector = nest.Create(
            'spike_detector', params=self.spike_detector_paramsR)
        nest.Connect(recall_neurons, recall_spike_detector)
        # save the detector
        self.sdR.append(recall_spike_detector)

    def disable_pattern_recall_setup(self, pattern_number):
        """Undo the setup."""
        recall_neurons = self.patterns[pattern_number - 1]
        nest.SetStatus(recall_neurons, {'I_e': 220.0})

    def recall_last_pattern(self, time, step):
        """Only setup the last pattern."""
        self.setup_pattern_for_recall(self.pattern_count - 1)
        self.run_simulation(time, step)
        self.disable_pattern_recall_setup(self.pattern_count - 1)

    def lesion_network(self):
        """Lesion the network."""

    def test_repair(self):
        """Let the network repair itself."""

if __name__ == "__main__":
    step = False
    stabilisation_time = 900
    simulation = Sinha2016()
    simulation.setup_simulation()
    simulation.run_simulation(stabilisation_time, step)

    # store and stabilise patterns
    for i in range(0, simulation.numpats):
        simulation.store_pattern()
        simulation.run_simulation(stabilisation_time, step)

    # Only recall the last pattern because nest doesn't do snapshots
    simulation.lesion_network()
    simulation.test_repair()
    simulation.recall_last_pattern(2, step)
    simulation.run_simulation(50, False)
