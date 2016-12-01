#!/usr/bin/env python
"""
NEST simulation code for my PhD research.

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
import sys
sys.argv.append('--quiet')
import nest
import numpy
import math
import random


class Sinha2016:

    """Simulations for my PhD 2016."""

    def __init__(self):
        """Initialise variables."""
        self.step = False
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1
        # time to stabilise network after pattern storage etc.
        self.stabilisation_time = 12000.  # seconds
        self.recording_interval = 500.  # seconds

        # plasticities
        self.structural_p = True
        self.synaptic_p = True

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'P': 800, 'R': 400,
                            'D': 200, 'STIM': 1000, 'Poisson': 1}

        # structural plasticity bits
        # must be an integer
        self.sp_update_interval = 100  # ms
        # time recall stimulus is enabled for
        self.recall_time = 1000.  # ms
        # Number of patterns we store
        self.numpats = 0

        self.recall_ext_i = 3000.

        self.rank = nest.Rank()

        self.patterns = []
        self.neuronsStim = []
        self.recalls = []
        self.sdP = []
        self.sdR = []
        self.sdL = []
        self.sdB = []
        self.sdStim = []
        self.pattern_spike_count_file_names = []
        self.pattern_spike_count_files = []
        self.pattern_count = 0

        self.weightEE = 3.
        self.weightII = -30.
        self.weightEI = 3.
        self.weightExtE = 50.
        self.weightExtI = 50.

        random.seed(42)

    def __setup_neurons(self):
        """Setup properties of neurons."""
        # if structural plasticity is enabled
        # Growth curves
        # eta is the minimum calcium concentration
        # epsilon is the target mean calcium concentration
        if self.structural_p:
            self.growth_curve_axonal_E = {
                'growth_curve': "gaussian",
                'growth_rate': 0.0001,  # Beta (elements/ms)
                'continuous': False,
                'eta': 0.4,
                'eps': 0.7,
            }
            self.growth_curve_axonal_I = {
                'growth_curve': "gaussian",
                'growth_rate': 0.0001,  # Beta (elements/ms)
                'continuous': False,
                'eta': 0.4,
                'eps': 0.7,
            }
            self.growth_curve_dendritic_E = {
                'growth_curve': "gaussian",
                'growth_rate': 0.0001,  # Beta (elements/ms)
                'continuous': False,
                'eta': 0.1,
                'eps': 0.7,
            }
            self.growth_curve_dendritic_I = {
                'growth_curve': "gaussian",
                'growth_rate': 0.0001,  # Beta (elements/ms)
                'continuous': False,
                'eta': 0.1,
                'eps': 0.7,
            }

            self.structural_p_elements_E = {
                'Den_ex': self.growth_curve_dendritic_E,
                'Den_in': self.growth_curve_dendritic_I,
                'Axon_ex': self.growth_curve_axonal_E
            }

            self.structural_p_elements_I = {
                'Den_ex': self.growth_curve_dendritic_I,
                'Den_in': self.growth_curve_dendritic_I,
                'Axon_in': self.growth_curve_axonal_I
            }

        # see the aif source for symbol definitions
        self.neuronDict = {'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.}
        # Set up TIF neurons
        # Setting up two models because then it makes it easier for me to get
        # them when I need to set up patterns
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults("tif_neuronE", self.neuronDict)
        nest.CopyModel("iaf_cond_exp", "tif_neuronI")
        nest.SetDefaults("tif_neuronI", self.neuronDict)

        # external current
        self.poissonExtDict = {'rate': 10., 'origin': 0., 'start': 0.}

    def __create_neurons(self):
        """Create our neurons."""
        if self.structural_p:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'], {
                'synaptic_elements': self.structural_p_elements_E})
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'], {
                'synaptic_elements': self.structural_p_elements_I})
        else:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        self.poissonExtE = nest.Create('poisson_generator',
                                       self.populations['Poisson'],
                                       params=self.poissonExtDict)
        self.poissonExtI = nest.Create('poisson_generator',
                                       self.populations['Poisson'],
                                       params=self.poissonExtDict)

    def __create_sparse_list(self, length, static_w, sparsity):
        """Create one list to use with SetStatus."""
        weights = []
        valid_values = int(length * sparsity)
        weights = (
            ([float(static_w), ] * valid_values) +
            ([0., ] * int(length - valid_values)))

        random.shuffle(weights)
        return weights

    def __fill_matrix(self, weightlist, static_w, sparsity):
        """Create a weight matrix to use in syn dict."""
        weights = []
        for row in weightlist:
            if isinstance(row, (list, tuple)):
                rowlen = len(row)
                valid_values = int(rowlen * sparsity)
                arow = (
                    ([float(static_w), ] * valid_values) +
                    ([0., ] * int(rowlen - valid_values))
                )
                random.shuffle(arow)
                weights.append(arow)
        return weights

    def __setup_matrix(self, pre_dim, post_dim, static_w, sparsity):
        """Create a weight matrix to use in syn dict."""
        weights = []
        valid_values = int(post_dim * sparsity)
        for i in range(0, pre_dim):
            arow = (
                ([float(static_w), ] * valid_values) +
                ([0., ] * int(post_dim - valid_values))
            )
            random.shuffle(arow)
            weights.append(arow)

        return weights

    def __get_synapses_to_form(self, sources, destinations, sparsity):
        """
        Find prospective synaptic connections between sets of neurons.

        Since structural plasticity does not permit sparse connections, I'm
        going to try to manually find the right number of syapses and connect
        neurons to get a certain sparsity.
        """
        possible_synapses = []
        required_synapses = int(float(len(sources)) * float(len(destinations))
                                * sparsity)
        chosen_synapses = []

        chosen_sources = numpy.random.choice(
            sources, size=required_synapses, replace=True)
        chosen_destinations = numpy.random.choice(
            destinations, size=required_synapses, replace=True)

        # There is a possibility of autapses, but given the high number of
        # options, it is unlikely. We'll also test multiple simulations and
        # average our results, so in reality, we'll get a sparsity of 2% on
        # average, which is OK.
        for i in range(0, required_synapses):
            chosen_synapses.append([chosen_sources[i], chosen_destinations[i]])

        return chosen_synapses

    def __setup_connections(self):
        """Setup connections."""
        # Global sparsity
        self.sparsity = 0.02
        self.sparsityStim = 0.05

        # Other connection numbers
        self.connectionNumberStim = int((self.populations['STIM'] *
                                         self.populations['R'])
                                        * self.sparsityStim)
        # From the butz paper
        self.connectionNumberExtE = 1
        self.connectionNumberExtI = 1

        # connection dictionaries
        self.connDictExtE = {'rule': 'fixed_indegree',
                             'indegree': self.connectionNumberExtE}
        self.connDictExtI = {'rule': 'fixed_indegree',
                             'indegree': self.connectionNumberExtI}
        self.connDictStim = {'rule': 'fixed_total_number',
                             'N': self.connectionNumberStim}

        # If neither, we've messed up
        if not self.structural_p and not self.synaptic_p:
            print("Neither plasticity is enabled. Exiting.")
            sys.exit()

        # Documentation says things are normalised in the iaf neuron so that
        # weight of 1 translates to 1nS
        # Only structural plasticity - if synapses are formed, give them
        # constant conductances
        if self.structural_p:
            if not self.synaptic_p:
                self.synDictEE = {'model': 'static_synapse',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictEI = {'model': 'static_synapse',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictII = {'model': 'static_synapse',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

                self.synDictIE = {'model': 'static_synapse',
                                  'weight': -1.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

            # both enabled
            else:
                self.synDictEE = {'model': 'static_synapse',
                                  'weight': self.weightEE,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictEI = {'model': 'static_synapse',
                                  'weight': self.weightEI,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictII = {'model': 'static_synapse',
                                  'weight': self.weightII,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

                self.synDictIE = {'model': 'vogels_sprekeler_synapse',
                                  'weight': -0.0000001, 'Wmax': -30000.,
                                  'alpha': .12, 'eta': 0.01,
                                  'tau': 20.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

        # Only synaptic plasticity - do not define synaptic elements
        else:
            self.synDictEE = {'model': 'static_synapse',
                              'weight': self.weightEE}
            self.synDictEI = {'model': 'static_synapse',
                              'weight': self.weightEI}

            self.synDictII = {'model': 'static_synapse',
                              'weight': self.weightII}

            self.synDictIE = {'model': 'vogels_sprekeler_synapse',
                              'weight': -0.0000001, 'Wmax': -30000.,
                              'alpha': .12, 'eta': 0.01,
                              'tau': 20.}

    def __connect_neurons(self):
        """Connect the neuron sets up."""
        nest.Connect(self.poissonExtE, self.neuronsE,
                     conn_spec=self.connDictExtE,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtE})
        nest.Connect(self.poissonExtI, self.neuronsI,
                     conn_spec=self.connDictExtI,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtI})

        # only structural plasticity
        if self.structural_p and not self.synaptic_p:
            print("Only structural plasticity enabled" +
                  "Not setting up any synapses.")
        # only synaptic plasticity
        # setup connections using Nest methods
        elif self.synaptic_p and not self.structural_p:
            conndict = {'rule': 'pairwise_bernoulli',
                        'p': self.sparsity}
            print("Setting up EE connections.")
            nest.Connect(self.neuronsE, self.neuronsE,
                         syn_spec=self.synDictEE,
                         conn_spec=conndict)
            print("EE connections set up.")

            print("Setting up EI connections.")
            nest.Connect(self.neuronsE, self.neuronsI,
                         syn_spec=self.synDictEI,
                         conn_spec=conndict)
            print("EI connections set up.")

            print("Setting up II connections.")
            nest.Connect(self.neuronsI, self.neuronsI,
                         syn_spec=self.synDictII,
                         conn_spec=conndict)
            print("II connections set up.")

            print("Setting up IE connections.")
            nest.Connect(self.neuronsI, self.neuronsE,
                         syn_spec=self.synDictIE,
                         conn_spec=conndict)
            print("IE connections set up.")
        # manually set up initial connections if structural plasticity and
        # synaptic plasticity are both enabled
        # This is because you can only either have all-all or one-one
        # connections when structural plasticity is enabled
        elif self.structural_p and self.synaptic_p:
            conndict = {'rule': 'one_to_one'}
            print("Setting up EE connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsE, self.neuronsE, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictEE,
                             conn_spec=conndict)
            print("{} EE connections set up.".format(
                len(synapses_to_create)))

            print("Setting up EI connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsE, self.neuronsI, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictEI,
                             conn_spec=conndict)
            print("{} EI connections set up.".format(
                len(synapses_to_create)))

            print("Setting up II connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsI, self.neuronsI, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictII,
                             conn_spec=conndict)
            print("{} II connections set up.".format(
                len(synapses_to_create)))

            print("Setting up IE connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsI, self.neuronsE, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictIE,
                             conn_spec=conndict)
            print("{} IE weights set up.".format(
                len(synapses_to_create)))

    def __setup_detectors(self):
        """Setup spike detectors."""
        self.spike_detector_paramsE = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-E'
        }
        self.spike_detector_paramsI = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-I'
        }
        self.spike_detector_paramsD = {
            'to_file': True,
            'label': 'spikes-' + str(self.rank) + '-deaffed'
        }

        self.sdE = nest.Create('spike_detector',
                               params=self.spike_detector_paramsE)
        self.sdI = nest.Create('spike_detector',
                               params=self.spike_detector_paramsI)

        nest.Connect(self.neuronsE, self.sdE)
        nest.Connect(self.neuronsI, self.sdI)

    def __setup_files(self):
        """Set up the filenames and handles."""
        # Get the number of spikes in these files and then post-process them to
        # get the firing rate and so on

        self.synaptic_p_weights_file_name_EE = (
            "00-synaptic-weights-EE-" + str(self.rank) + ".txt")
        self.weights_file_handle_EE = open(
            self.synaptic_p_weights_file_name_EE, 'w')
        print("{},{}".format(
            "time(ms)", "EE(nS)"),
            file=self.weights_file_handle_EE)

        self.synaptic_p_weights_file_name_EI = (
            "00-synaptic-weights-EI-" + str(self.rank) + ".txt")
        self.weights_file_handle_EI = open(
            self.synaptic_p_weights_file_name_EI, 'w')
        print("{},{}".format(
            "time(ms)", "EI(nS)"),
            file=self.weights_file_handle_EI)

        self.synaptic_p_weights_file_name_II = (
            "00-synaptic-weights-II-" + str(self.rank) + ".txt")
        self.weights_file_handle_II = open(
            self.synaptic_p_weights_file_name_II, 'w')
        print("{},{}".format(
            "time(ms)", "II(nS)"),
            file=self.weights_file_handle_II)

        self.synaptic_p_weights_file_name_IE = (
            "00-synaptic-weights-IE-" + str(self.rank) + ".txt")
        self.weights_file_handle_IE = open(
            self.synaptic_p_weights_file_name_IE, 'w')
        print("{},{}".format(
            "time(ms)", "IE(nS)"),
            file=self.weights_file_handle_IE)

        self.ca_filename_E = ("01-calcium-E-" +
                              str(self.rank) + ".txt")
        self.ca_file_handle_E = open(self.ca_filename_E, 'w')
        print("{}, {}".format(
            "time(ms)", "cal_E values"), file=self.ca_file_handle_E)

        self.ca_filename_I = ("01-calcium-I-" +
                              str(self.rank) + ".txt")
        self.ca_file_handle_I = open(self.ca_filename_I, 'w')
        print("{}, {}".format(
            "time(ms)", "cal_I values"), file=self.ca_file_handle_I)

        if self.structural_p:
            self.syn_elms_filename_E = ("02-synaptic-elements-totals-E-" +
                                        str(self.rank) + ".txt")
            self.syn_elms_file_handle_E = open(self.syn_elms_filename_E, 'w')
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format
                (
                    "time(ms)",
                    "a_ex_total", "a_ex_connected",
                    "d_ex_ex_total", "d_ex_ex_connected",
                    "d_ex_in_total", "d_ex_in_connected",
                ),
                file=self.syn_elms_file_handle_E)

            self.syn_elms_filename_I = ("02-synaptic-elements-totals-I-" +
                                        str(self.rank) + ".txt")
            self.syn_elms_file_handle_I = open(self.syn_elms_filename_I, 'w')
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format
                (
                    "time(ms)",
                    "a_in_total", "a_in_connected",
                    "d_in_ex_total", "d_in_ex_connected",
                    "d_in_in_total", "d_in_in_connected"
                ),
                file=self.syn_elms_file_handle_I)

    def setup_plasticity(self, structural_p=True, synaptic_p=True):
        """Control plasticities."""
        self.structural_p = structural_p
        self.synaptic_p = synaptic_p

        if self.structural_p and self.synaptic_p:
            print("BOTH PLASTICITIES ENABLED")
        elif self.structural_p and not self.synaptic_p:
            print("ONLY STRUCTURAL PLASTICITY ENABLED")
        elif self.synaptic_p and not self.structural_p:
            print("ONLY SYNAPTIC PLASTICITY ENABLED")
        else:
            print("Both plasticities cannot be disabled. Exiting.")
            sys.exit()

    def setup_test_simulation(self, step=None,
                              stabilisation_time=None,
                              recording_interval=None):
        """Set up a test simulation."""
        if step:
            self.step = step
        if stabilisation_time:
            self.stabilisation_time = stabilisation_time
        if recording_interval:
            self.recording_interval = recording_interval

        # Nest stuff
        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_DEBUG')

        # a much smaller simulation
        self.populations = {'E': 80, 'I': 20, 'P': 8, 'R': 4,
                            'D': 2, 'STIM': 10, 'Poisson': 1}

        self.__setup_simulation()

    def setup_simulation(self, step=False,
                         stabilisation_time=None,
                         recording_interval=None):
        """Set up non test simulation."""
        if step:
            self.step = step
        if stabilisation_time:
            self.stabilisation_time = stabilisation_time
        if recording_interval:
            self.recording_interval = recording_interval

        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_INFO')

        self.__setup_simulation()

    def __setup_simulation(self):
        """Setup the common simulation things."""
        # Nest stuff
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
        # Update the SP interval
        if self.structural_p:
            nest.EnableStructuralPlasticity()
            nest.SetStructuralPlasticityStatus({
                'structural_plasticity_update_interval':
                self.sp_update_interval,
            })

        self.__setup_neurons()
        self.__create_neurons()
        self.__setup_detectors()

        self.__setup_connections()
        self.__connect_neurons()

        self.__setup_files()

        self.dump_data()

    def stabilise(self):
        """Stabilise network."""
        sim_steps = numpy.arange(0, self.stabilisation_time,
                                 self.recording_interval)
        for i, j in enumerate(sim_steps):
            self.run_simulation(self.recording_interval)

    def run_simulation(self, simtime=2000):
        """Run the simulation."""
        sim_steps = numpy.arange(0, simtime)
        if self.step:
            print("Stepping through the simulation one second at a time")
            for i, step in enumerate(sim_steps):

                nest.Simulate(1000)
                self.__dump_synaptic_weights()
        else:
            print("Not stepping through it one second at a time")
            nest.Simulate(simtime*1000)
            self.dump_data()

            current_simtime = (
                str(nest.GetKernelStatus()['time']) + "msec")
            print("Simulation time: " "{}".format(current_simtime))

    def store_pattern(self):
        """ Store a pattern and set up spike detectors."""
        spike_detector_paramsP = {
            'to_file': True,
            'label': ('spikes-' + str(self.rank) + '-pattern-' +
                      str(self.pattern_count))
        }
        spike_detector_paramsB = {
            'to_file': True,
            'label': ('spikes-' + str(self.rank) + '-background-' +
                      str(self.pattern_count))
        }

        local_neurons = nest.GetNodes(
            nest.CurrentSubnet(), {'model': 'tif_neuronE'},
            local_only=False)[0]

        pattern_neurons = random.sample(
            local_neurons,
            self.populations['P'])
        print("ANKUR>> Number of pattern neurons: "
              "{}".format(len(pattern_neurons)))

        # strengthen connections
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        print("ANKUR>> Number of connections strengthened: "
              "{}".format(len(connections)))
        nest.SetStatus(connections, {"weight": 24.})

        # store these neurons
        self.patterns.append(pattern_neurons)
        # print to file
        file_name = "patternneurons-{}-rank-{}.txt".format(self.pattern_count,
                                                           self.rank)
        file_handle = open(file_name, 'w')
        for neuron in pattern_neurons:
            print(neuron, file=file_handle)
        file_handle.close()

        # background neurons
        background_neurons = list(set(local_neurons) - set(pattern_neurons))
        file_name = "backgroundneurons-{}-rank-{}.txt".format(
            self.pattern_count, self.rank)

        file_handle = open(file_name, 'w')
        for neuron in background_neurons:
            print(neuron, file=file_handle)
        file_handle.close()

        # set up spike detectors
        # pattern
        pattern_spike_detector = nest.Create(
            'spike_detector', params=spike_detector_paramsP)
        nest.Connect(pattern_neurons, pattern_spike_detector)
        # save the detector
        self.sdP.append(pattern_spike_detector)

        # background
        background_spike_detector = nest.Create(
            'spike_detector', params=spike_detector_paramsB)
        nest.Connect(background_neurons, background_spike_detector)
        # save the detector
        self.sdB.append(background_spike_detector)

        # Increment count after it's all been set up
        self.pattern_count += 1
        print("Number of patterns stored: {}".format(self.pattern_count))

    def setup_pattern_for_recall(self, pattern_number):
        """
        Set up a pattern for recall.

        Creates a new poisson generator and connects it to a recall subset of
        this pattern - the poisson stimulus will run for the set recall_time
        from the invocation of this method.
        """
        # set up external stimulus
        stim_time = nest.GetKernelStatus()['time']
        neuronDictStim = {'rate': 200.,
                          'origin': stim_time,
                          'start': 0., 'stop': self.recall_time}
        spike_detector_paramsStim = {
            'to_file': True,
            'label': ('spikes-' + str(self.rank) + '-Stim-' +
                      str(pattern_number))

        }

        stim = nest.Create('poisson_generator', 1,
                           neuronDictStim)
        stim_neurons = nest.Create('parrot_neuron',
                                   self.populations['STIM'])
        nest.Connect(stim, stim_neurons)
        sd = nest.Create('spike_detector',
                         params=spike_detector_paramsStim)
        nest.Connect(stim_neurons, sd)
        self.sdStim.append(sd)
        self.neuronsStim.append(stim_neurons)

        pattern_neurons = self.patterns[pattern_number - 1]
        recall_neurons = random.sample(
            pattern_neurons,
            self.populations['R'])
        print("ANKUR>> Number of recall neurons: "
              "{}".format(len(recall_neurons)))

        nest.Connect(stim_neurons, recall_neurons,
                     conn_spec=self.connDictStim)

        self.recalls.append(recall_neurons)

        # print to file
        file_name = "recallneurons-{}-rank-{}.txt".format(self.pattern_count,
                                                          self.rank)
        file_handle = open(file_name, 'w')
        for neuron in recall_neurons:
            print(neuron, file=file_handle)
        file_handle.close()

        spike_detector_paramsR = {
            'to_file': True,
            'label': ('spikes-' + str(self.rank) + '-recall-' +
                      str(self.pattern_count))
        }
        recall_spike_detector = nest.Create(
            'spike_detector', params=spike_detector_paramsR)
        nest.Connect(recall_neurons, recall_spike_detector)
        # save the detector
        self.sdR.append(recall_spike_detector)
        # nest.SetStatus(recall_neurons, {'I_e': self.recall_ext_i})

    def recall_last_pattern(self, time):
        """
        Only setup the last pattern.

        An extra helper method, since we'll be doing this most.
        """
        self.recall_pattern(time, self.pattern_count)

    def recall_pattern(self, time, pattern_number):
        """Recall a pattern."""
        self.setup_pattern_for_recall(pattern_number)
        self.run_simulation(time)

    def deaff_last_pattern(self):
        """
        Deaff last pattern.

        An extra helper method, since we'll be doing this most.
        """
        self.deaff_pattern(self.pattern_count - 1)

    def deaff_pattern(self, pattern_number):
        """Deaff the network."""
        pattern_neurons = self.patterns[pattern_number - 1]
        deaffed_neurons = random.sample(
            pattern_neurons,
            self.populations['D'])
        print("ANKUR>> Number of deaff neurons: "
              "{}".format(len(deaffed_neurons)))
        nest.SetStatus(deaffed_neurons, {'I_e': 0.})

        deaff_spike_detector = nest.Create(
            'spike_detector', params=self.spike_detector_paramsD)
        nest.Connect(deaffed_neurons, deaff_spike_detector)
        # save the detector
        self.sdL.append(deaff_spike_detector)

    def __dump_ca_concentration(self):
        """Dump calcium concentration."""
        loc_e = [stat['global_id'] for stat in nest.GetStatus(self.neuronsE)
                 if stat['local']]
        loc_i = [stat['global_id'] for stat in nest.GetStatus(self.neuronsI)
                 if stat['local']]
        ca_e = nest.GetStatus(loc_e, 'Ca')
        ca_i = nest.GetStatus(loc_i, 'Ca')

        current_simtime = (str(nest.GetKernelStatus()['time']))
        print("{}, {}".format(current_simtime,
                              str(ca_e).strip('[]').strip('()')),
              file=self.ca_file_handle_E)

        print("{}, {}".format(current_simtime,
                              str(ca_i).strip('[]').strip('()')),
              file=self.ca_file_handle_I)

    def __dump_synaptic_elements_per_neurons(self):
        """
        Dump synaptic elements for each neuron for a time.

        neuronid    ax_total    ax_connected    den_ex_total ...
        """
        if self.structural_p:
            loc_e = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsE)
                     if stat['local']]
            loc_i = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsI)
                     if stat['local']]

            current_simtime = (str(nest.GetKernelStatus()['time']))

            synaptic_element_file_E = (
                "03-synaptic-elements-E-" + str(self.rank) + "-" +
                current_simtime + ".txt")
            with open(synaptic_element_file_E, 'w') as filehandle_E:
                print("neuronID\tAxon_ex\tAxon_ex_connected" +
                      "\tDend_ex\tDend_ex_con\t" +
                      "Dend_in\tDend_in_con", file=filehandle_E)

                for neuron in loc_e:
                    syn_elms = nest.GetStatus([neuron], 'synaptic_elements')[0]
                    axons = syn_elms['Axon_ex']['z']
                    axons_conn = syn_elms['Axon_ex']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        neuron,
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn
                    ), file=filehandle_E)

            synaptic_element_file_I = (
                "03-synaptic-elements-I-" + str(self.rank) + "-" +
                current_simtime + ".txt")
            with open(synaptic_element_file_I, 'w') as filehandle_I:
                print("neuronID\tAxon_in\tAxon_in_connected" +
                      "\tDend_ex\tDend_ex_con\t" +
                      "Dend_in\tDend_in_con", file=filehandle_I)

                for neuron in loc_i:
                    syn_elms = nest.GetStatus([neuron], 'synaptic_elements')[0]
                    axons = syn_elms['Axon_in']['z']
                    axons_conn = syn_elms['Axon_in']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        neuron,
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn
                    ), file=filehandle_I)

    def __dump_total_synaptic_elements(self):
        """Dump total number of synaptic elements."""
        if self.structural_p:
            loc_e = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsE)
                     if stat['local']]
            loc_i = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsI)
                     if stat['local']]
            syn_elems_e = nest.GetStatus(loc_e, 'synaptic_elements')
            syn_elems_i = nest.GetStatus(loc_i, 'synaptic_elements')

            current_simtime = (str(nest.GetKernelStatus()['time']))

            # Only need presynaptic elements to find number of synapses
            # Excitatory neuron set
            axons_ex_total = sum(neuron['Axon_ex']['z'] for neuron in
                                 syn_elems_e)
            axons_ex_connected = sum(neuron['Axon_ex']['z_connected']
                                     for neuron in syn_elems_e)
            dendrites_ex_ex_total = sum(neuron['Den_ex']['z'] for neuron in
                                        syn_elems_e)
            dendrites_ex_ex_connected = sum(neuron['Den_ex']['z_connected'] for
                                            neuron in syn_elems_e)
            dendrites_ex_in_total = sum(neuron['Den_in']['z'] for neuron in
                                        syn_elems_e)
            dendrites_ex_in_connected = sum(neuron['Den_in']['z_connected'] for
                                            neuron in syn_elems_e)

            # Inhibitory neuron set
            axons_in_total = sum(neuron['Axon_in']['z'] for neuron in
                                 syn_elems_i)
            axons_in_connected = sum(neuron['Axon_in']['z_connected']
                                     for neuron in syn_elems_i)
            dendrites_in_ex_total = sum(neuron['Den_ex']['z'] for neuron in
                                        syn_elems_i)
            dendrites_in_ex_connected = sum(neuron['Den_ex']['z_connected'] for
                                            neuron in syn_elems_i)
            dendrites_in_in_total = sum(neuron['Den_in']['z'] for neuron in
                                        syn_elems_i)
            dendrites_in_in_connected = sum(neuron['Den_in']['z_connected'] for
                                            neuron in syn_elems_i)

            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format
                (
                    current_simtime,
                    axons_ex_total, axons_ex_connected,
                    dendrites_ex_ex_total, dendrites_ex_ex_connected,
                    dendrites_ex_in_total, dendrites_ex_in_connected,
                ),
                file=self.syn_elms_file_handle_E)

            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format
                (
                    current_simtime,
                    axons_in_total, axons_in_connected,
                    dendrites_in_ex_total, dendrites_in_ex_connected,
                    dendrites_in_in_total, dendrites_in_in_connected,
                ),
                file=self.syn_elms_file_handle_I)

    def __dump_synaptic_weights(self):
        """Dump synaptic weights."""
        current_simtime = (str(nest.GetKernelStatus()['time']))

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsI)
        weightsIE = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsIE).strip('[]').strip('()')),
            file=self.weights_file_handle_IE)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsI)
        weightsII = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsII).strip('[]').strip('()')),
            file=self.weights_file_handle_II)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsE)
        weightsEI = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEI).strip('[]').strip('()')),
            file=self.weights_file_handle_EI)

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsE)
        weightsEE = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEE).strip('[]').strip('()')),
            file=self.weights_file_handle_EE)

    def dump_data(self):
        """Master datadump function."""
        self.__dump_synaptic_weights()
        self.__dump_ca_concentration()
        self.__dump_synaptic_elements_per_neurons()
        self.__dump_total_synaptic_elements()


if __name__ == "__main__":
    step = False
    test = False
    simulation = Sinha2016()

    # Enable plasticities
    simulation.setup_plasticity(True, True)

    if test:
        simulation.setup_test_simulation(
            stabilisation_time=200., recording_interval=10.)
        simulation.stabilise()
    else:
        simulation.setup_simulation(
            stabilisation_time=2000., recording_interval=200.)
        simulation.stabilise()

        # store and stabilise patterns
        for i in range(0, simulation.numpats):
            simulation.store_pattern()
            simulation.stabilise()

        # Only recall the last pattern because nest doesn't do snapshots
        # simulation.deaff_last_pattern()
        # simulation.stabilise()
        # simulation.recall_last_pattern(50)
