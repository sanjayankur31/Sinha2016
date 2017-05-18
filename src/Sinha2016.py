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
# use random.sample instead of numpy.random - faster
import random
from scipy.spatial import cKDTree
from mpi4py import MPI


class Sinha2016:

    """Simulations for my PhD 2016."""

    def __init__(self):
        """Initialise variables."""
        self.comm = MPI.COMM_WORLD
        self.step = False
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1
        # time to stabilise network after pattern storage etc.
        self.stabilisation_time = 12000.  # seconds
        # keep this a divisor of the structural plasticity update interval, and
        # the stabilisation time for simplicity
        self.recording_interval = 500.  # seconds

        # what plasticity should the network be setup to handle
        self.setup_str_p = True
        self.setup_syn_p = True
        self.rewiring_enabled = False

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'STIM': 1000, 'Poisson': 1}
        # pattern percent of E neurons
        self.pattern_percent = .1
        # recall percent of pattern
        self.recall_percent = .25

        # without spatial information
        # deafferentation percent of pattern
        self.deaff_random_pattern_percent = .50
        # deafferentation percent of background
        self.deaff_bg_random_percentE = .50
        self.deaff_bg_random_percentI = .50

        self.populations['P'] = self.pattern_percent * self.populations['E']
        self.populations['R'] = self.recall_percent * self.populations['P']

        # location bits
        self.colsE = 80
        self.colsI = 40
        self.neuronal_distE = 150  # micro metres
        self.neuronal_distI = 300  # micro metres
        self.location_sd = 15  # micro metres
        self.location_tree = None

        # structural plasticity bits
        # not steps since we're not using it in NEST. This is for our manual
        # updates
        self.sp_update_interval = 1000.  # seconds
        # time recall stimulus is enabled for
        self.recall_time = 1000.  # ms

        self.recall_ext_i = 3000.

        self.rank = nest.Rank()

        self.patterns = []
        self.neuronsStim = []
        self.recalls = []
        self.sdP = []
        self.sdR = []
        self.sdDP = []
        self.sdDBG_E = []
        self.sdDBG_I = []
        self.sdB = []
        self.sdStim = []
        self.pattern_spike_count_file_names = []
        self.pattern_spike_count_files = []
        self.pattern_count = 0

        self.wbar = 0.5
        self.weightEE = self.wbar
        self.weightII = self.wbar * -10.
        self.weightEI = self.wbar
        self.weightPatternEE = self.wbar * 5.
        self.weightExtE = 50.
        self.weightExtI = self.weightExtE

        random.seed(42)

        # used to track how many comma separated values each line will have
        # when I store synaptic conductances.
        # Required in post processing, so that I know what the size of my
        # dataframe should be. Pandas cannot figure this out on its own. See
        # postprocessing scripts for more information.
        self.num_synapses_EE = 0
        self.num_synapses_EI = 0
        self.num_synapses_II = 0
        self.num_synapses_IE = 0

    def __setup_neurons(self):
        """Setup properties of neurons."""
        # if structural plasticity is enabled
        # Growth curves
        # eta is the minimum calcium concentration
        # epsilon is the target mean calcium concentration
        if self.setup_str_p:
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
                           'tau_syn_ex': 5., 'tau_syn_in': 10.,
                           'beta_Ca': 0.012
                           }
        # Set up TIF neurons
        # Setting up two models because then it makes it easier for me to get
        # them when I need to set up patterns
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults("tif_neuronE", self.neuronDict)
        nest.CopyModel("iaf_cond_exp", "tif_neuronI")
        nest.SetDefaults("tif_neuronI", self.neuronDict)

        # external stimulus
        # if synaptic plasticity is enabled, we have an initial set of
        # connections, so we don't need so many connections
        if self.setup_syn_p:
            self.poissonExtDict = {'rate': 10., 'origin': 0., 'start': 0.}
        # else, if no synaptic plasticity, only structural, so we need more
        # input stimulus to get the connections to form
        else:
            self.poissonExtDict = {'rate': 50., 'origin': 0., 'start': 0.}

    def __create_neurons(self):
        """Create our neurons."""
        if self.setup_str_p:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'], {
                'synaptic_elements': self.structural_p_elements_E})
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'], {
                'synaptic_elements': self.structural_p_elements_I})
        else:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        # Generate a grid and construct a cKDTree
        locations = []
        for neuron in self.neuronsE:
            y = random.gauss(
                int((neuron - self.neuronsE[0])/self.colsE) *
                self.neuronal_distE, self.location_sd)
            x = random.gauss(
                ((neuron - self.neuronsE[0]) % self.colsE) *
                self.neuronal_distE, self.location_sd)
            locations.append([x, y])
        # I neurons have an intiail offset to distribute them evenly between E
        # neurons
        for neuron in self.neuronsI:
            y = self.neuronal_distI/4 + random.gauss(
                int((neuron - self.neuronsI[0])/self.colsI) *
                self.neuronal_distI, self.location_sd)
            x = self.neuronal_distI/4 + random.gauss(
                ((neuron - self.neuronsI[0]) % self.colsI) *
                self.neuronal_distI, self.location_sd)
            locations.append([x, y])

        self.location_tree = cKDTree(locations)

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
        required_synapses = int(float(len(sources)) * float(len(destinations))
                                * sparsity)
        chosen_synapses = []

        # native python random.choices is quicker but isn't in py < 3.6
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

    def __setup_initial_connection_params(self):
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
        if not self.setup_str_p and not self.setup_syn_p:
            print("Neither plasticity is enabled. Exiting.")
            sys.exit()

        # Documentation says things are normalised in the iaf neuron so that
        # weight of 1 translates to 1nS
        # Only structural plasticity - if synapses are formed, give them
        # constant conductances
        nest.CopyModel('static_synapse', 'static_synapse_ex')
        nest.CopyModel('static_synapse', 'static_synapse_in')
        if self.setup_str_p:
            if not self.setup_syn_p:
                self.synDictEE = {'model': 'static_synapse_ex',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictEI = {'model': 'static_synapse_ex',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictII = {'model': 'static_synapse_in',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

                self.synDictIE = {'model': 'static_synapse_in',
                                  'weight': -1.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

            # both enabled
            else:
                self.synDictEE = {'model': 'static_synapse_ex',
                                  'weight': self.weightEE,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictEI = {'model': 'static_synapse_ex',
                                  'weight': self.weightEI,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}

                self.synDictII = {'model': 'static_synapse_in',
                                  'weight': self.weightII,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

                self.synDictIE = {'model': 'vogels_sprekeler_synapse',
                                  'weight': -0.0000001, 'Wmax': -30000.,
                                  'alpha': .12, 'eta': 0.01,
                                  'tau': 20.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}

            nest.SetStructuralPlasticityStatus({
                'structural_plasticity_synapses': {
                    'synapseEE': self.synDictEE,
                    'synapseEI': self.synDictEI,
                    'synapseII': self.synDictII,
                    'synapseIE': self.synDictIE,
                }
            })

        # Only synaptic plasticity - do not define synaptic elements
        else:
            self.synDictEE = {'model': 'static_synapse_ex',
                              'weight': self.weightEE}
            self.synDictEI = {'model': 'static_synapse_ex',
                              'weight': self.weightEI}

            self.synDictII = {'model': 'static_synapse_in',
                              'weight': self.weightII}

            self.synDictIE = {'model': 'vogels_sprekeler_synapse',
                              'weight': -0.0000001, 'Wmax': -30000.,
                              'alpha': .12, 'eta': 0.01,
                              'tau': 20.}

    def __create_initial_connections(self):
        """Initially connect various neuron sets."""
        nest.Connect(self.poissonExtE, self.neuronsE,
                     conn_spec=self.connDictExtE,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtE})
        nest.Connect(self.poissonExtI, self.neuronsI,
                     conn_spec=self.connDictExtI,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtI})

        # only structural plasticity
        if self.setup_str_p and not self.setup_syn_p:
            print("Only structural plasticity enabled" +
                  "Not setting up any synapses.")
        # only synaptic plasticity
        # setup connections using Nest methods
        elif self.setup_syn_p and not self.setup_str_p:
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
        elif self.setup_str_p and self.setup_syn_p:
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
            'label': 'spikes-E'
        }
        self.spike_detector_paramsI = {
            'to_file': True,
            'label': 'spikes-I'
        }
        self.spike_detector_paramsP = {
            'to_file': True,
            'label': 'spikes-pattern'
        }
        self.spike_detector_paramsB = {
            'to_file': True,
            'label': 'spikes-background'
        }
        self.spike_detector_paramsR = {
            'to_file': True,
            'label': 'spikes-recall'
        }
        self.spike_detector_paramsDP = {
            'to_file': True,
            'label': 'spikes-deaffed-pattern'
        }
        self.spike_detector_paramsDBG_E = {
            'to_file': True,
            'label': 'spikes-deaffed-bg-E'
        }
        self.spike_detector_paramsDBG_I = {
            'to_file': True,
            'label': 'spikes-deaffed-bg-I'
        }
        self.spike_detector_paramsStim = {
            'to_file': True,
            'label': 'spikes-stim'
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

        if self.setup_str_p:
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
        self.setup_str_p = structural_p
        self.setup_syn_p = synaptic_p

        if self.setup_str_p and self.setup_syn_p:
            print("NETWORK SETUP TO HANDLE BOTH PLASTICITIES")
        elif self.setup_str_p and not self.setup_syn_p:
            print("NETWORK SETUP TO HANDLE ONLY STRUCTURAL PLASTICITY")
        elif self.setup_syn_p and not self.setup_str_p:
            print("NETWORK SETUP TO HANDLE ONLY SYNAPTIC PLASTICITY")
        else:
            print("Both plasticities cannot be disabled. Exiting.")
            sys.exit()

    def prerun_setup(self, step=False,
                     stabilisation_time=None,
                     sp_update_interval=None,
                     recording_interval=None):
        """Pre reun configuration."""
        # Cannot be changed mid simulation
        if step:
            self.step = step
        self.update_windows(stabilisation_time, sp_update_interval,
                            recording_interval)
        self.__setup_simulation()

    def update_windows(self,
                       stabilisation_time=None,
                       sp_update_interval=None,
                       recording_interval=None):
        """Set up stabilisation time."""
        if stabilisation_time:
            self.stabilisation_time = stabilisation_time
        if sp_update_interval:
            self.sp_update_interval = sp_update_interval
        if recording_interval:
            self.recording_interval = recording_interval

    def __setup_simulation(self):
        """Setup the common simulation things."""
        # Nest stuff
        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_INFO')

        nest.SetKernelStatus(
            {
                'resolution': self.dt,
                'local_num_threads': 1,
                'overwrite_files': True
            }
        )
        # Since I've patched NEST, this doesn't actually update connectivity
        # But, it's required to ensure that synaptic elements are connected
        # correctly when I form or delete new connections
        nest.EnableStructuralPlasticity()
        nest.SetStructuralPlasticityStatus({
            'structural_plasticity_update_interval':
            int(self.sp_update_interval),
        })

        self.__setup_neurons()
        self.__create_neurons()
        self.__setup_detectors()

        self.__setup_initial_connection_params()
        self.__create_initial_connections()

        self.__setup_files()

        nest.Prepare()

        self.dump_data()

    def stabilise(self):
        """Stabilise network."""
        print("SIMULATION: STABILISING for {} seconds".format(
            self.stabilisation_time))
        update_steps = numpy.arange(0, self.stabilisation_time,
                                    self.sp_update_interval)
        for i, j in enumerate(update_steps):
            sim_steps = numpy.arange(0, self.sp_update_interval,
                                     self.recording_interval)
            for j, k in enumerate(sim_steps):
                self.run_simulation(self.recording_interval)
            self.update_connectivity()

    def run_simulation(self, simtime=2000):
        """Run the simulation."""
        if self.step:
            sim_steps = numpy.arange(0, simtime)
            for i, step in enumerate(sim_steps):
                nest.Run(1000)
                self.__dump_synaptic_weights()
        else:
            nest.Run(simtime*1000)
            self.dump_data()
            current_simtime = (
                str(nest.GetKernelStatus()['time']) + "msec")
            print("Simulation time: " "{}".format(current_simtime))

    def __get_syn_elms(self):
        """Get synaptic elements all neurons."""
        # Holds the deltas, not the actual numbers
        # A dictionary - the key is the neuron's gid and the value is another
        # dictionary that contains the deltas
        synaptic_elms = {}
        # must get local neurons, since only local neurons will have global_id
        # and synaptic_element values in their dicts. The rest are proxies.
        local_neurons = [stat['global_id'] for stat in
                         nest.GetStatus(self.neuronsE + self.neuronsI) if
                         stat['local']]

        lneurons = nest.GetStatus(local_neurons, ['global_id',
                                                  'synaptic_elements'])
        # returns a list of sets - one set from each rank
        ranksets = self.comm.allgather(lneurons)

        for rankset in ranksets:
            for neuron in rankset:
                gid = neuron[0]
                synelms = neuron[1]

                if 'Axon_in' in synelms:
                    source_elms_con = synelms['Axon_in']['z_connected']
                    source_elms_total = synelms['Axon_in']['z']
                elif 'Axon_ex' in synelms:
                    source_elms_con = synelms['Axon_ex']['z_connected']
                    source_elms_total = synelms['Axon_ex']['z']

                target_elms_con_ex = synelms['Den_ex']['z_connected']
                target_elms_con_in = synelms['Den_in']['z_connected']
                target_elms_total_ex = synelms['Den_ex']['z']
                target_elms_total_in = synelms['Den_in']['z']
                delta_z_ax = (math.floor(source_elms_total) -
                              source_elms_con)
                delta_z_d_ex = (math.floor(target_elms_total_ex) -
                                target_elms_con_ex)
                delta_z_d_in = (math.floor(target_elms_total_in) -
                                target_elms_con_in)

                if 'Axon_in' in synelms:
                    elms = {
                        'Axon_in': (delta_z_ax),
                        'Den_ex': (delta_z_d_ex),
                        'Den_in': (delta_z_d_in),
                    }
                    synaptic_elms[gid] = elms
                elif 'Axon_ex' in synelms:
                    elms = {
                        'Axon_ex': (delta_z_ax),
                        'Den_ex': (delta_z_d_ex),
                        'Den_in': (delta_z_d_in),
                    }
                    synaptic_elms[gid] = elms

        return synaptic_elms

    def __delete_random_connections(self, synelms):
        """Delete connections randomly."""
        print("Deleting RANDOM connections")
        # the order in which these are removed should not matter - whether we
        # remove connections using axons first or dendrites first, the end
        # state of the network should be the same.
        # Note that we are modifying a dictionary while iterating over it. This
        # is OK here since we're not modifying the keys, only the values.
        # http://stackoverflow.com/a/2315529/375067
        for nrn in synelms.iteritems():
            # excitatory neurons as sources
            gid = nrn[0]
            elms = nrn[1]
            if 'Axon_ex' in elms and elms['Axon_ex'] < 0.0:
                conns = nest.GetConnections(source=[gid],
                                            synapse_model='static_synapse_ex')
                targets = []
                chosen_targets = []

                for acon in conns:
                    targets.append(acon[1])

                if len(targets) > 0:
                    # this is where the selection logic is
                    if len(targets) > int(abs(elms['Axon_ex'])):
                        chosen_targets = random.sample(
                            targets, int(abs(elms['Axon_ex'])))
                    else:
                        chosen_targets = targets

                    nest.Disconnect(
                        pre=[gid], post=chosen_targets, syn_spec={
                            'model': 'static_synapse_ex',
                            'pre_synaptic_element': 'Axon_ex',
                            'post_synaptic_element': 'Den_ex',
                        }, conn_spec={
                            'rule': 'all_to_all'}
                    )
                    elms['Axon_ex'] += len(chosen_targets)
                    for t in chosen_targets:
                        synelms[t]['Den_ex'] += 1

            # inhibitory neurons as sources
            elif 'Axon_in' in elms and elms['Axon_in'] < 0.0:
                conns = nest.GetConnections(source=[gid],
                                            synapse_model='static_synapse_in')
                targets = []
                chosen_targets = []

                for acon in conns:
                    targets.append(acon[1])

                if len(targets) > 0:
                    # this is where the selection logic is
                    if len(targets) > int(abs(elms['Axon_in'])):
                        chosen_targets = random.sample(
                            targets, int(abs(elms['Axon_in'])))
                    else:
                        chosen_targets = targets

                    nest.Disconnect(
                        pre=[gid], post=chosen_targets, syn_spec={
                            'model': 'static_synapse',
                            'pre_synaptic_element': 'Axon_in',
                            'post_synaptic_element': 'Den_in',
                        }, conn_spec={
                            'rule': 'all_to_all'}
                    )
                    elms['Axon_in'] += len(chosen_targets)
                    for t in chosen_targets:
                        synelms[t]['Den_in'] += 1

            # excitatory dendrites as targets
            if 'Den_ex' in elms and elms['Den_ex'] < 0.0:
                conns = nest.GetConnections(target=[gid],
                                            synapse_model='static_synapse_ex')
                sources = []
                chosen_sources = []

                for acon in conns:
                    sources.append(acon[0])

                if len(sources) > 0:
                    if len(sources) > int(abs(elms['Den_ex'])):
                        chosen_sources = random.sample(
                            sources, int(abs(elms['Den_ex'])))
                    else:
                        chosen_sources = sources

                    nest.Disconnect(
                        pre=chosen_sources, post=[gid], syn_spec={
                            'model': 'static_synapse_ex',
                            'pre_synaptic_element': 'Axon_ex',
                            'post_synaptic_element': 'Den_ex',
                        }, conn_spec={
                            'rule': 'all_to_all'}
                    )
                    elms['Den_ex'] += len(chosen_sources)
                    for s in chosen_sources:
                        synelms[s]['Axon_ex'] += 1

            # inhibitory dendrites as targets
            if 'Den_in' in elms and elms['Den_in'] < 0.0:
                conns = nest.GetConnections(target=[gid],
                                            synapse_model='static_synapse_in')
                sources = []
                chosen_sources = []

                for acon in conns:
                    sources.append(acon[0])

                if len(sources) > 0:
                    if len(sources) > int(abs(elms['Den_in'])):
                        chosen_sources = random.sample(
                            sources, int(abs(elms['Den_in'])))
                    else:
                        chosen_sources = sources

                    nest.Disconnect(
                        pre=chosen_sources, post=[gid], syn_spec={
                            'model': 'static_synapse_in',
                            'pre_synaptic_element': 'Axon_in',
                            'post_synaptic_element': 'Den_in',
                        }, conn_spec={
                            'rule': 'all_to_all'}
                    )
                    elms['Den_in'] += len(chosen_sources)
                    for s in chosen_sources:
                        synelms[s]['Axon_in'] += 1

    def __create_random_connections(self, synelms):
        """Connect random neurons to create new connections."""
        print("Creating RANDOM connections")
        for nrn in synelms.iteritems():
            gid = nrn[0]
            elms = nrn[1]
            # excitatory connections - only need to look at Axons, it doesn't
            # matter which synaptic elements you start with, whichever are less
            # will act as the limiting factor.
            if 'Axon_ex' in elms and elms['Axon_ex'] > 0.0:
                targets = []
                chosen_targets = []

                for atarget in synelms.iteritems():
                    tid = atarget[0]
                    telms = atarget[1]
                    if 'Den_ex' in telms and telms['Den_ex'] > 0.0:
                        # add the target multiple times, since it has multiple
                        # available contact points
                        targets.extend([tid]*int(telms['Den_ex']))

                if len(targets) > 0:
                    if len(targets) > int(abs(elms['Axon_ex'])):
                        chosen_targets = random.sample(
                            targets, int(abs(elms['Axon_ex'])))
                    else:
                        chosen_targets = targets

                    nest.Connect([gid], chosen_targets,
                                 conn_spec='all_to_all',
                                 syn_spec={'model': 'static_synapse_ex',
                                           'pre_synaptic_element': 'Axon_ex',
                                           'post_synaptic_element': 'Den_ex'
                                           })
                    for cho in chosen_targets:
                        synelms[cho]['Den_ex'] -= 1
                    elms['Axon_ex'] -= len(chosen_targets)

            if 'Axon_in' in elms and elms['Axon_in'] > 0.0:
                targets = []
                chosen_targets = []

                for atarget in synelms.iteritems():
                    tid = atarget[0]
                    telms = atarget[1]
                    if 'Den_in' in telms and telms['Den_in'] > 0.0:
                        # add the target multiple times, since it has multiple
                        # available contact points
                        targets.intend([tid]*int(telms['Den_in']))

                if len(targets) > 0:
                    if len(targets) > int(abs(elms['Axon_in'])):
                        chosen_targets = random.sample(
                            targets, int(abs(elms['Axon_in'])))
                    else:
                        chosen_targets = targets

                    nest.Connect([gid], chosen_targets,
                                 conn_spec='all_to_all',
                                 syn_spec={'model': 'static_synapse_in',
                                           'pre_synaptic_element': 'Axon_in',
                                           'post_synaptic_element': 'Den_in'
                                           })
                    for cho in chosen_targets:
                        synelms[cho]['Den_in'] -= 1
                    elms['Axon_in'] -= len(chosen_targets)

    def update_connectivity(self):
        """Our implementation of structural plasticity."""
        if not self.rewiring_enabled:
            return
        syn_elms = self.__get_syn_elms()
        with open('synelms-{}.txt'.format(nest.Rank()), 'w') as f:
            print(syn_elms, file=f)

        self.__delete_random_connections(syn_elms)
        syn_elms = self.__get_syn_elms()
        self.__create_random_connections(syn_elms)
        nest.Prepare()

    def store_spatial_pattern(self, track=False):
        """Store a pattern of neurons that are spatially adjacent."""

    def store_random_pattern(self, track=False):
        """Store a pattern of neurons that are randomly chosen."""
        print("SIMULATION: Storing pattern {}".format(self.pattern_count + 1))
        # Keep track of how many patterns are stored
        self.pattern_count += 1
        pattern_neurons = random.sample(
            self.neuronsE, int(math.ceil(len(self.neuronsE) *
                                         self.pattern_percent)))
        print("ANKUR>> Number of pattern neurons: "
              "{}".format(len(pattern_neurons)))

        # strengthen connections
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        print("ANKUR>> Number of connections strengthened: "
              "{}".format(len(connections)))
        nest.SetStatus(connections, {"weight": self.weightPatternEE})
        print("ANKUR>> New weight: {}nS".format(self.weightPatternEE))

        # store these neurons
        self.patterns.append(pattern_neurons)
        if track:
            # print to file
            file_name = "patternneurons-{}-rank-{}.txt".format(
                self.pattern_count, self.rank)
            with open(file_name, 'w') as file_handle:
                for neuron in pattern_neurons:
                    print(neuron, file=file_handle)

            # background neurons
            background_neurons = list(
                set(self.neuronsE) - set(pattern_neurons))
            file_name = "backgroundneurons-{}-rank-{}.txt".format(
                self.pattern_count, self.rank)

            with open(file_name, 'w') as file_handle:
                for neuron in background_neurons:
                    print(neuron, file=file_handle)

            # set up spike detectors
            sd_params = self.spike_detector_paramsP.copy()
            sd_params['label'] = (sd_params['label'] + "-{}".format(
                self.pattern_count))
            # pattern
            pattern_spike_detector = nest.Create(
                'spike_detector', params=sd_params)
            nest.Connect(pattern_neurons, pattern_spike_detector)
            # save the detector
            self.sdP.append(pattern_spike_detector)

            # background
            sd_params = self.spike_detector_paramsB.copy()
            sd_params['label'] = (sd_params['label'] + "-{}".format(
                self.pattern_count))
            background_spike_detector = nest.Create(
                'spike_detector', params=sd_params)
            nest.Connect(background_neurons, background_spike_detector)
            # save the detector
            self.sdB.append(background_spike_detector)

        print("Number of patterns stored: {}".format(self.pattern_count))
        nest.Prepare()

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

        sd_params = self.spike_detector_paramsStim.copy()
        sd_params['label'] = sd_params['label'] + "-{}".format(pattern_number)

        stim = nest.Create('poisson_generator', 1,
                           neuronDictStim)
        # TODO: Do I need a parrot neuron?
        stim_neurons = nest.Create('parrot_neuron',
                                   self.populations['STIM'])
        nest.Connect(stim, stim_neurons)
        sd = nest.Create('spike_detector',
                         params=sd_params)
        nest.Connect(stim_neurons, sd)
        self.sdStim.append(sd)
        self.neuronsStim.append(stim_neurons)

        pattern_neurons = self.patterns[pattern_number - 1]
        recall_neurons = random.sample(
            pattern_neurons, int(math.ceil(len(pattern_neurons) *
                                           self.recall_percent)))
        print("ANKUR>> Number of recall neurons: "
              "{}".format(len(recall_neurons)))

        nest.Connect(stim_neurons, recall_neurons,
                     conn_spec=self.connDictStim)

        self.recalls.append(recall_neurons)

        # print to file
        file_name = "recallneurons-{}-rank-{}.txt".format(pattern_number,
                                                          self.rank)
        with open(file_name, 'w') as file_handle:
            for neuron in recall_neurons:
                print(neuron, file=file_handle)

        sd_params = self.spike_detector_paramsR.copy()
        sd_params['label'] = sd_params['label'] + "-{}".format(pattern_number)
        recall_spike_detector = nest.Create(
            'spike_detector', params=sd_params)
        nest.Connect(recall_neurons, recall_spike_detector)
        # save the detector
        self.sdR.append(recall_spike_detector)
        nest.Prepare()

    def recall_last_pattern(self, time):
        """
        Only setup the last pattern.

        An extra helper method, since we'll be doing this most.
        """
        print("SIMULATION: RECALLING LAST PATTERN")
        self.recall_pattern(time, self.pattern_count)

    def recall_pattern(self, time, pattern_number):
        """Recall a pattern."""
        self.setup_pattern_for_recall(pattern_number)
        self.run_simulation(time)

    def deaff_last_random_pattern(self):
        """
        Deaff last pattern by picking a random set of neurons from it.

        An extra helper method, since we'll be doing this most.
        """
        print("SIMULATION: deaffing last pattern ({})".format(
            self.pattern_count))
        self.__deaff_random_pattern(self.pattern_count)
        self.__deaff_bg_random_E(self.pattern_count)
        self.__deaff_bg_random_I(self.pattern_count)
        nest.Prepare()

    def deaff_random_pattern(self, pattern_number):
        """Deaff a pattern by picking a random set of neurons from it."""
        self.__deaff_random_pattern(pattern_number)
        self.__deaff_bg_random_E(pattern_number)
        self.__deaff_bg_random_I(pattern_number)

    def __deaff_random_pattern(self, pattern_number):
        """Deaff the pattern neuron set by picking a random set of neurons."""
        print("ANKUR>> Deaffing pattern {}".format(pattern_number))
        pattern_neurons = self.patterns[pattern_number - 1]
        deaffed_neurons = random.sample(
            pattern_neurons, int(math.ceil(len(pattern_neurons) *
                                           self.deaff_random_pattern_percent)))
        print("ANKUR>> Number of deaff pattern neurons: "
              "{}".format(len(deaffed_neurons)))
        if len(deaffed_neurons) > 0:
            conns = nest.GetConnections(source=self.poissonExtE,
                                        target=deaffed_neurons)
            for conn in conns:
                nest.DisconnectOneToOne(conn[0], conn[1],
                                        syn_spec={'model': 'static_synapse'})

            sd_params = self.spike_detector_paramsDP.copy()
            sd_params['label'] = sd_params['label'] + "-{}".format(
                pattern_number)
            deaff_spike_detector = nest.Create(
                'spike_detector', params=sd_params)
            nest.Connect(deaffed_neurons, deaff_spike_detector)
            # save the detector
            self.sdDP.append(deaff_spike_detector)

            file_name = "deaffed-patternneurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            self.__dump_neuron_set(file_name, deaffed_neurons)

            file_name = "non-deaffed-patternneurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            non_deaffed_neurons = list(set(pattern_neurons) -
                                       set(deaffed_neurons))
            self.__dump_neuron_set(file_name, non_deaffed_neurons)

    def __deaff_bg_random_E(self, pattern_number):
        """Deaff background a random selection of E neurons."""
        pattern_neurons = self.patterns[pattern_number - 1]
        bg_neurons = list(set(self.neuronsE) - set(pattern_neurons))
        deaffed_neurons = random.sample(
            bg_neurons, int(math.ceil(len(bg_neurons) *
                                      self.deaff_bg_random_percentE)))
        print("ANKUR>> Number of deaff bg E neurons: "
              "{}".format(len(deaffed_neurons)))
        if len(deaffed_neurons) > 0:
            conns = nest.GetConnections(source=self.poissonExtE,
                                        target=deaffed_neurons)
            for conn in conns:
                nest.DisconnectOneToOne(conn[0], conn[1],
                                        syn_spec={'model': 'static_synapse'})

            sd_params = self.spike_detector_paramsDBG_E.copy()
            sd_params['label'] = sd_params['label'] + "-{}".format(
                pattern_number)
            deaff_spike_detector = nest.Create(
                'spike_detector', params=sd_params)
            nest.Connect(deaffed_neurons, deaff_spike_detector)
            # save the detector
            self.sdDBG_E.append(deaff_spike_detector)

            file_name = "deaffed-backgroundneurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            self.__dump_neuron_set(file_name, deaffed_neurons)

            file_name = "non-deaffed-backgroundneurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            non_deaffed_neurons = list(set(bg_neurons) -
                                       set(deaffed_neurons))
            self.__dump_neuron_set(file_name, non_deaffed_neurons)

    def __deaff_bg_random_I(self, pattern_number):
        """Deaff a random selection of background I neurons."""
        deaffed_neurons = random.sample(
            self.neuronsI, int(math.ceil(len(self.neuronsI) *
                                         self.deaff_bg_random_percentI)))

        print("ANKUR>> Number of deaff bg I neurons: "
              "{}".format(len(deaffed_neurons)))
        if len(deaffed_neurons) > 0:
            conns = nest.GetConnections(source=self.poissonExtI,
                                        target=deaffed_neurons)
            for conn in conns:
                nest.DisconnectOneToOne(conn[0], conn[1],
                                        syn_spec={'model': 'static_synapse'})

            sd_params = self.spike_detector_paramsDBG_I.copy()
            sd_params['label'] = sd_params['label'] + "-{}".format(
                pattern_number)
            deaff_spike_detector = nest.Create(
                'spike_detector', params=sd_params)
            nest.Connect(deaffed_neurons, deaff_spike_detector)
            # save the detector
            self.sdDBG_I.append(deaff_spike_detector)

            file_name = "deaffed-Ineurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            self.__dump_neuron_set(file_name, deaffed_neurons)

            file_name = "non-deaffed-Ineurons-{}-rank-{}.txt".format(
                pattern_number, self.rank)
            non_deaffed_neurons = list(set(self.neuronsI) -
                                       set(deaffed_neurons))
            self.__dump_neuron_set(file_name, non_deaffed_neurons)

    def __dump_neuron_set(self, file_name, neurons):
        """Dump a set of neuronIDs to a text file."""
        with open(file_name, 'w') as file_handle:
            for neuron in neurons:
                print(neuron, file=file_handle)

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
        if self.setup_str_p:
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
        if self.setup_str_p:
            loc_e = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsE)
                     if stat['local']]
            loc_i = [stat['global_id'] for stat
                     in nest.GetStatus(self.neuronsI)
                     if stat['local']]
            syn_elms_e = nest.GetStatus(loc_e, 'synaptic_elements')
            syn_elms_i = nest.GetStatus(loc_i, 'synaptic_elements')

            current_simtime = (str(nest.GetKernelStatus()['time']))

            # Only need presynaptic elements to find number of synapses
            # Excitatory neuron set
            axons_ex_total = sum(neuron['Axon_ex']['z'] for neuron in
                                 syn_elms_e)
            axons_ex_connected = sum(neuron['Axon_ex']['z_connected']
                                     for neuron in syn_elms_e)
            dendrites_ex_ex_total = sum(neuron['Den_ex']['z'] for neuron in
                                        syn_elms_e)
            dendrites_ex_ex_connected = sum(neuron['Den_ex']['z_connected'] for
                                            neuron in syn_elms_e)
            dendrites_ex_in_total = sum(neuron['Den_in']['z'] for neuron in
                                        syn_elms_e)
            dendrites_ex_in_connected = sum(neuron['Den_in']['z_connected'] for
                                            neuron in syn_elms_e)

            # Inhibitory neuron set
            axons_in_total = sum(neuron['Axon_in']['z'] for neuron in
                                 syn_elms_i)
            axons_in_connected = sum(neuron['Axon_in']['z_connected']
                                     for neuron in syn_elms_i)
            dendrites_in_ex_total = sum(neuron['Den_ex']['z'] for neuron in
                                        syn_elms_i)
            dendrites_in_ex_connected = sum(neuron['Den_ex']['z_connected'] for
                                            neuron in syn_elms_i)
            dendrites_in_in_total = sum(neuron['Den_in']['z'] for neuron in
                                        syn_elms_i)
            dendrites_in_in_connected = sum(neuron['Den_in']['z_connected'] for
                                            neuron in syn_elms_i)

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
        if len(weightsIE) > self.num_synapses_IE:
            self.num_synapses_IE = len(weightsIE)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsI)
        weightsII = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsII).strip('[]').strip('()')),
            file=self.weights_file_handle_II)
        if len(weightsII) > self.num_synapses_II:
            self.num_synapses_II = len(weightsII)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsE)
        weightsEI = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEI).strip('[]').strip('()')),
            file=self.weights_file_handle_EI)
        if len(weightsEI) > self.num_synapses_EI:
            self.num_synapses_EI = len(weightsEI)

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsE)
        weightsEE = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEE).strip('[]').strip('()')),
            file=self.weights_file_handle_EE)
        if len(weightsEE) > self.num_synapses_EE:
            self.num_synapses_EE = len(weightsEE)

    def dump_data(self):
        """Master datadump function."""
        self.__dump_synaptic_weights()
        self.__dump_ca_concentration()
        self.__dump_synaptic_elements_per_neurons()
        self.__dump_total_synaptic_elements()

    def close_files(self):
        """Close all files when the simulation is finished."""
        # Comma printed so that pandas can read it as a dataframe point
        print("{},".format(self.num_synapses_EE),
              file=self.weights_file_handle_EE)
        self.weights_file_handle_EE.close()

        print("{},".format(self.num_synapses_EI),
              file=self.weights_file_handle_EI)
        self.weights_file_handle_EI.close()
        print("{},".format(self.num_synapses_II),
              file=self.weights_file_handle_II)
        self.weights_file_handle_II.close()

        print("{},".format(self.num_synapses_IE),
              file=self.weights_file_handle_IE)
        self.weights_file_handle_IE.close()

        local_neurons = [stat['global_id'] for stat in
                         nest.GetStatus(self.neuronsE) if stat['local']]
        print("{},".format(len(local_neurons)), file=self.ca_file_handle_E)
        self.ca_file_handle_E.close()
        local_neurons = [stat['global_id'] for stat in
                         nest.GetStatus(self.neuronsI) if stat['local']]
        print("{},".format(len(local_neurons)), file=self.ca_file_handle_I)
        self.ca_file_handle_I.close()

        if self.setup_str_p:
            self.syn_elms_file_handle_E.close()
            self.syn_elms_file_handle_I.close()

    def enable_rewiring(self):
        """Enable the rewiring."""
        self.rewiring_enabled = True

    def disable_rewiring(self):
        """Disable the rewiring."""
        self.rewiring_enabled = False

if __name__ == "__main__":
    step = False
    numpats = 1
    simulation = Sinha2016()

    # Setup network to handle plasticities
    # update of the network
    print("SIMULATION STARTED")
    simulation.setup_plasticity(True, True)

    # Intial stabilisation #
    simulation.prerun_setup(
        stabilisation_time=2000.,
        sp_update_interval=1000.,
        recording_interval=200.)
    simulation.stabilise()

    # Pattern related simulation
    if numpats > 0:
        # store patterns
        # only track first pattern to limit log files
        simulation.store_random_pattern(True)
        # Do not track the others
        for i in range(1, numpats):
            simulation.store_random_pattern()

        # stabilise network after storing patterns
        simulation.stabilise()

        # Deaff first pattern (which is also being tracked)
        simulation.deaff_random_pattern(1)
        # Enable structural plasticity for repair #
        simulation.enable_rewiring()
        # Stabilise for repair
        simulation.stabilise()
        simulation.stabilise()

        # recall stored and tracked pattern
        simulation.recall_pattern(50, 1)

    simulation.close_files()
    nest.Cleanup()
    print("SIMULATION FINISHED SUCCESSFULLY")
