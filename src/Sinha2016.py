#!/usr/bin/env python3
"""
NEST simulation code for my PhD research.

File: Sinha2016.py

Copyright 2017 Ankur Sinha
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

import sys
import nest
import numpy
import math
# use random.sample instead of numpy.random - faster
import random
from scipy.spatial import cKDTree
from mpi4py import MPI
import logging
import operator


class Sinha2016:

    """Simulations for my PhD 2016."""

    def __init__(self):
        """Initialise variables."""
        self.comm = MPI.COMM_WORLD
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1
        # time to stabilise network after pattern storage etc.
        self.default_stabilisation_time = 12000.  # seconds
        # keep this a divisor of the structural plasticity update interval, and
        # the stabilisation time for simplicity
        self.recording_interval = 500.  # seconds

        # what plasticity should the network be setup to handle
        self.is_str_p_enabled = True
        self.is_syn_p_enabled = True
        self.is_rewiring_enabled = False
        self.is_metaplasticity_enabled = True
        # "random" or "distance" or "weight"
        self.syn_del_strategy = "random"
        # "random" or "distance"
        self.syn_form_strategy = "distance"

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'STIM': 1000, 'Poisson': 1}
        # pattern percent of E neurons
        self.pattern_percent = .1
        # recall percent of pattern
        self.recall_percent = .25

        self.populations['P'] = self.pattern_percent * self.populations['E']
        self.populations['R'] = self.recall_percent * self.populations['P']

        # location bits
        self.colsE = 80
        self.colsI = 40
        self.neuronal_distE = 150  # micro metres
        self.neuronal_distI = 300  # micro metres
        self.location_sd = 15  # micro metres
        self.location_tree = None
        self.lpz_percent = 0.5

        # structural plasticity bits
        # not steps since we're not using it in NEST. This is for our manual
        # updates
        self.sp_update_interval = 1000.  # seconds
        # time recall stimulus is enabled for
        self.recall_duration = 1000.  # ms
        # homoeostatic stable points for E and I neurons
        # eps is for a neuron - same for axonal and dendritic elements
        # but eta is different for axonal and dendritic elements
        self.eps_e = 0.7
        self.eps_i = 0.7
        self.eta_ax_e = 0.4
        self.eta_ax_i = 0.4
        self.eta_den_e = 0.1
        self.eta_den_i = 0.1
        self.nu = 0.0001  # maximum value of dz/dt

        self.rank = nest.Rank()

        self.patterns = []
        self.recall_neurons = []
        self.sdP = []
        self.sdB = []
        self.pattern_spike_count_fns = []
        self.pattern_spike_count_files = []
        self.pattern_count = 0

        self.wbar = 0.5
        self.weightEE = self.wbar
        self.weightII = self.wbar * -10.
        self.weightEI = self.wbar  # is the same as EE, specified for clarity
        self.weightPatternEE = self.wbar * 5.
        self.weightExt = 50.

        # used to track how many comma separated values each line will have
        # when I store synaptic conductances.
        # Required in post processing, so that I know what the size of my
        # dataframe should be. Pandas cannot figure this out on its own. See
        # postprocessing scripts for more information.
        self.num_synapses_EE = 0
        self.num_synapses_EI = 0
        self.num_synapses_II = 0
        self.num_synapses_IE = 0

        self.neuronsE = []
        self.lpz_c_neurons_E = []
        self.lpz_b_neurons_E = []
        self.lpz_neurons_E = []
        self.p_lpz_neurons_E = []
        self.o_neurons_E = []

        self.neuronsI = []
        self.lpz_c_neurons_I = []
        self.lpz_b_neurons_I = []
        self.lpz_neurons_I = []
        self.p_lpz_neurons_I = []
        self.o_neurons_I = []

        random.seed(42)

    def __setup_neurons(self):
        """Setup properties of neurons."""
        # if structural plasticity is enabled
        # Growth curves
        # eta is the minimum calcium concentration
        # epsilon is the target mean calcium concentration
        if self.is_str_p_enabled:
            # set all growth rates to zero initially so that no change in z
            # takes place. So, we have a stable network with the required
            # firing rate, the required synaptic connections, the required
            # numbers of synaptic elements, and we obtain the required values
            # of epsilon too.
            self.growth_curve_axonal_E = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'continuous': False,
                'eta': self.eta_ax_e,
                'eps': self.eps_e
            }
            self.growth_curve_axonal_I = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'continuous': False,
                'eta': self.eta_ax_i,
                'eps': self.eps_i
            }
            self.growth_curve_dendritic_E = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'continuous': False,
                'eta': self.eta_den_e,
                'eps': self.eps_e
            }
            self.growth_curve_dendritic_I = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'continuous': False,
                'eta': self.eta_den_i,
                'eps': self.eps_i
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
                           'beta_Ca': 0.05, 'tau_Ca': 10000.
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
        if self.is_syn_p_enabled:
            self.poissonExtDict = {'rate': 10., 'origin': 0., 'start': 0.}
        # else, if no synaptic plasticity, only structural, so we need more
        # input stimulus to get the connections to form
        else:
            self.poissonExtDict = {'rate': 50., 'origin': 0., 'start': 0.}

    def __create_neurons(self):
        """Create our neurons."""
        if self.is_str_p_enabled:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'], {
                'synaptic_elements': self.structural_p_elements_E})
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'], {
                'synaptic_elements': self.structural_p_elements_I})
        else:
            self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
            self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        # Generate a grid and construct a cKDTree
        locations = []
        if self.rank == 0:
            loc_file = open("00-locations-E.txt", 'w')
            print("gid\tcol\trow\txcor\tycor", file=loc_file, flush=True)
        for neuron in self.neuronsE:
            row = int((neuron - self.neuronsE[0])/self.colsE)
            y = random.gauss(row * self.neuronal_distE, self.location_sd)
            col = ((neuron - self.neuronsE[0]) % self.colsE)
            x = random.gauss(col * self.neuronal_distE, self.location_sd)
            locations.append([x, y])
            if self.rank == 0:
                print("{}\t{}\t{}\t{}\t{}".format(neuron, col, row, x, y),
                      file=loc_file, flush=True)
        if self.rank == 0:
            loc_file.close()

        # I neurons have an intiail offset to distribute them evenly between E
        # neurons
        if self.rank == 0:
            loc_file = open("00-locations-I.txt", 'w')
            print("gid\tcol\trow\txcor\tycor", file=loc_file, flush=True)
        for neuron in self.neuronsI:
            row = int((neuron - self.neuronsI[0])/self.colsI)
            y = self.neuronal_distI/4 + random.gauss(
                row * self.neuronal_distI, self.location_sd)
            col = ((neuron - self.neuronsI[0]) % self.colsI)
            x = self.neuronal_distI/4 + random.gauss(
                col * self.neuronal_distI, self.location_sd)
            locations.append([x, y])
            if self.rank == 0:
                print("{}\t{}\t{}\t{}\t{}".format(neuron, col, row, x, y),
                      file=loc_file, flush=True)
        if self.rank == 0:
            loc_file.close()
        self.location_tree = cKDTree(locations)

        self.poissonExt = nest.Create('poisson_generator',
                                      self.populations['Poisson'],
                                      params=self.poissonExtDict)

    def __get_regions(self):
        """Divide neurons into regions."""
        first_point = self.location_tree.data[0]
        last_point = self.location_tree.data[len(self.neuronsE) - 1]

        # lpz
        lpz_neurons = self.__get_neurons_from_region(
            (len(self.neuronsE) + len(self.neuronsI)) * self.lpz_percent,
            first_point, last_point)
        # centre of lpz
        lpz_c_neurons = self.__get_neurons_from_region(
            (len(self.neuronsE) + len(self.neuronsI)) * self.lpz_percent/2.,
            first_point, last_point)
        # so inner border of lpz
        lpz_b_neurons = list(set(lpz_neurons) - set(lpz_c_neurons))

        # lpz and the outer peri
        with_p_lpz_neurons = self.__get_neurons_from_region(
            (len(self.neuronsE) + len(self.neuronsI)) * self.lpz_percent * 2.,
            first_point, last_point)

        # for E
        self.lpz_c_neurons_E = list(set(lpz_c_neurons).intersection(
            set(self.neuronsE)))
        self.lpz_b_neurons_E = list(set(lpz_b_neurons).intersection(
            set(self.neuronsE)))
        self.lpz_neurons_E = (self.lpz_c_neurons_E +
                              self.lpz_b_neurons_E)
        with_p_lpz_neurons_E = list(set(with_p_lpz_neurons).intersection(
            set(self.neuronsE)))
        self.p_lpz_neurons_E = list(set(with_p_lpz_neurons_E) -
                                    set(lpz_neurons))
        # other E neurons that are not in lpz and p_lpz
        self.o_neurons_E = list(set(self.neuronsE - set(with_p_lpz_neurons_E)))

        # for I
        self.lpz_c_neurons_I = list(set(lpz_c_neurons).intersection(
            set(self.neuronsI)))
        self.lpz_b_neurons_I = list(set(lpz_b_neurons).intersection(
            set(self.neuronsI)))
        self.lpz_neurons_I = (self.lpz_c_neurons_I +
                              self.lpz_b_neurons_I)
        with_p_lpz_neurons_I = list(set(with_p_lpz_neurons).intersection(
            set(self.neuronsI)))
        self.p_lpz_neurons_I = list(set(with_p_lpz_neurons_I) -
                                    set(lpz_neurons))
        # other I neurons that are not in lpz and p_lpz
        self.o_neurons_I = list(set(self.neuronsI - set(with_p_lpz_neurons_I)))

        if self.rank == 0:
            # excitatory neurons
            with open("00-locations-o_E.txt", 'w') as f1:
                print("gid\txcor\tycor", file=f1, flush=True)
                for neuron in self.o_neurons_E:
                    nrnindex = neuron - self.neuronsE[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f1)
            with open("00-locations-p_lpz_E.txt", 'w') as f1:
                print("gid\txcor\tycor", file=f1, flush=True)
                for neuron in self.p_lpz_neurons_E:
                    nrnindex = neuron - self.neuronsE[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f1)
            with open("00-locations-lpz_c_E.txt", 'w') as f2:
                print("gid\txcor\tycor", file=f2, flush=True)
                for neuron in self.lpz_c_neurons_E:
                    nrnindex = neuron - self.neuronsE[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f2)
            with open("00-locations-lpz_b_E.txt", 'w') as f3:
                print("gid\txcor\tycor", file=f3, flush=True)
                for neuron in self.lpz_b_neurons_E:
                    nrnindex = neuron - self.neuronsE[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f3)

            with open("00-locations-o_I.txt", 'w') as f1:
                print("gid\txcor\tycor", file=f1, flush=True)
                for neuron in self.o_neurons_I:
                    nrnindex = neuron + self.neuronsE[-1] - self.neuronsI[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f1)
            with open("00-locations-p_lpz_I.txt", 'w') as f1:
                print("gid\txcor\tycor", file=f1, flush=True)
                for neuron in self.p_lpz_neurons_I:
                    nrnindex = neuron + self.neuronsE[-1] - self.neuronsI[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f1)
            with open("00-locations-lpz_c_I.txt", 'w') as f2:
                print("gid\txcor\tycor", file=f2, flush=True)
                for neuron in self.lpz_c_neurons_I:
                    nrnindex = neuron + self.neuronsE[-1] - self.neuronsI[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f2)
            with open("00-locations-lpz_b_I.txt", 'w') as f3:
                print("gid\txcor\tycor", file=f3, flush=True)
                for neuron in self.lpz_b_neurons_I:
                    nrnindex = neuron + self.neuronsE[-1] - self.neuronsI[0]
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[nrnindex][0],
                        self.location_tree.data[nrnindex][1]),
                        file=f3)

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
                                         self.populations['R']) *
                                        self.sparsityStim)
        # each neuron gets a single input
        self.connDictExt = {'rule': 'fixed_indegree',
                            'indegree': 1}
        # recall stimulus
        self.connDictStim = {'rule': 'fixed_total_number',
                             'N': self.connectionNumberStim}

        # If neither, we've messed up
        if not self.is_str_p_enabled and not self.is_syn_p_enabled:
            logging.critical("Neither plasticity is enabled. Exiting.")
            sys.exit()

        # Documentation says things are normalised in the iaf neuron so that
        # weight of 1 translates to 1nS
        # Only structural plasticity - if synapses are formed, give them
        # constant conductances
        nest.CopyModel('static_synapse', 'static_synapse_ex')
        nest.CopyModel('static_synapse', 'static_synapse_in')
        nest.CopyModel('vogels_sprekeler_synapse', 'stdp_synapse_in')
        if self.is_str_p_enabled:
            if not self.is_syn_p_enabled:
                self.synDictEE = {'model': 'static_synapse_ex',
                                  'weight': 1.,
                                  'pre_synaptic_element': 'Axon_ex',
                                  'post_synaptic_element': 'Den_ex'}
                self.synDictEI = self.synDictEE
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
                self.synDictEI = self.synDictEE
                self.synDictII = {'model': 'static_synapse_in',
                                  'weight': self.weightII,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}
                self.synDictIE = {'model': 'stdp_synapse_in',
                                  'weight': -0.0000001, 'Wmax': -30000.,
                                  'alpha': .12, 'eta': 0.01,
                                  'tau': 20.,
                                  'pre_synaptic_element': 'Axon_in',
                                  'post_synaptic_element': 'Den_in'}
            nest.SetStructuralPlasticityStatus({
                'structural_plasticity_synapses': {
                    'static_synapse_ex': self.synDictEE,
                    'static_synapse_in': self.synDictII,
                    'stdp_synapse_in': self.synDictIE,
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
            self.synDictIE = {'model': 'stdp_synapse_in',
                              'weight': -0.0000001, 'Wmax': -30000.,
                              'alpha': .12, 'eta': 0.01,
                              'tau': 20.}

    def __create_initial_connections(self):
        """Initially connect various neuron sets."""
        nest.Connect(self.poissonExt, self.neuronsE,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})
        nest.Connect(self.poissonExt, self.neuronsI,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})

        # only structural plasticity
        if self.is_str_p_enabled and not self.is_syn_p_enabled:
            logging.debug("Only structural plasticity enabled" +
                          "Not setting up any synapses.")
        # only synaptic plasticity
        # setup connections using Nest methods
        elif self.is_syn_p_enabled and not self.is_str_p_enabled:
            conndict = {'rule': 'pairwise_bernoulli',
                        'p': self.sparsity}
            logging.debug("Setting up EE connections.")
            nest.Connect(self.neuronsE, self.neuronsE,
                         syn_spec=self.synDictEE,
                         conn_spec=conndict)
            logging.debug("EE connections set up.")

            logging.debug("Setting up EI connections.")
            nest.Connect(self.neuronsE, self.neuronsI,
                         syn_spec=self.synDictEI,
                         conn_spec=conndict)
            logging.debug("EI connections set up.")

            logging.debug("Setting up II connections.")
            nest.Connect(self.neuronsI, self.neuronsI,
                         syn_spec=self.synDictII,
                         conn_spec=conndict)
            logging.debug("II connections set up.")

            logging.debug("Setting up IE connections.")
            nest.Connect(self.neuronsI, self.neuronsE,
                         syn_spec=self.synDictIE,
                         conn_spec=conndict)
            logging.debug("IE connections set up.")
        # manually set up initial connections if structural plasticity and
        # synaptic plasticity are both enabled
        # This is because you can only either have all-all or one-one
        # connections when structural plasticity is enabled
        elif self.is_str_p_enabled and self.is_syn_p_enabled:
            conndict = {'rule': 'one_to_one'}
            logging.debug("Setting up EE connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsE, self.neuronsE, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictEE,
                             conn_spec=conndict)
            logging.debug("{} EE connections set up.".format(
                len(synapses_to_create)))

            logging.debug("Setting up EI connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsE, self.neuronsI, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictEI,
                             conn_spec=conndict)
            logging.debug("{} EI connections set up.".format(
                len(synapses_to_create)))

            logging.debug("Setting up II connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsI, self.neuronsI, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictII,
                             conn_spec=conndict)
            logging.debug("{} II connections set up.".format(
                len(synapses_to_create)))

            logging.debug("Setting up IE connections.")
            synapses_to_create = self.__get_synapses_to_form(
                self.neuronsI, self.neuronsE, self.sparsity)
            for source, destination in synapses_to_create:
                nest.Connect([source], [destination],
                             syn_spec=self.synDictIE,
                             conn_spec=conndict)
            logging.debug("{} IE weights set up.".format(
                len(synapses_to_create)))

    def __setup_detectors(self):
        """Setup spike detectors."""
        # E neurons
        self.sd_params_lpz_c_E = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-lpz_c_E'
        }
        self.sd_params_lpz_b_E = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-lpz_b_E'
        }
        self.sd_params_p_lpz_E = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-p_lpz_E'
        }
        self.sd_params_o_E = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-o_E'
        }
        self.sd_params_lpz_c_I = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-lpz_c_I'
        }
        self.sd_params_lpz_b_I = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-lpz_b_I'
        }
        self.sd_params_p_lpz_I = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-p_lpz_I'
        }
        self.sd_params_o_I = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-o_I'
        }
        # pattern neurons
        self.sd_paramsP = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-pattern'
        }
        # background neurons
        self.sd_paramsB = {
            'to_file': True,
            'to_memory': False,
            'label': 'spikes-background'
        }

        self.sd_lpz_c_E = nest.Create('spike_detector',
                                      params=self.sd_params_lpz_c_E)
        self.sd_lpz_b_E = nest.Create('spike_detector',
                                      params=self.sd_params_lpz_b_E)
        self.sd_p_lpz_E = nest.Create('spike_detector',
                                      params=self.sd_params_p_lpz_E)
        self.sd_o_E = nest.Create('spike_detector',
                                  params=self.sd_params_o_E)
        self.sd_lpz_c_I = nest.Create('spike_detector',
                                      params=self.sd_params_lpz_c_I)
        self.sd_lpz_b_I = nest.Create('spike_detector',
                                      params=self.sd_params_lpz_b_I)
        self.sd_p_lpz_I = nest.Create('spike_detector',
                                      params=self.sd_params_p_lpz_I)
        self.sd_o_I = nest.Create('spike_detector',
                                  params=self.sd_params_o_I)

        nest.Connect(self.lpz_c_neurons_E, self.sd_lpz_c_E)
        nest.Connect(self.lpz_b_neurons_E, self.sd_lpz_b_E)
        nest.Connect(self.p_lpz_neurons_E, self.sd_p_lpz_E)
        nest.Connect(self.o_neurons_E, self.sd_o_E)
        nest.Connect(self.lpz_c_neurons_I, self.sd_lpz_c_I)
        nest.Connect(self.lpz_b_neurons_I, self.sd_lpz_b_I)
        nest.Connect(self.p_lpz_neurons_I, self.sd_p_lpz_I)
        nest.Connect(self.o_neurons_I, self.sd_o_I)

    def __setup_files(self):
        """Set up the filenames and handles."""
        self.synaptic_p_weights_fn_EE = (
            "01-synaptic-weights-EE-" + str(self.rank) + ".txt")
        self.weights_fh_EE = open(
            self.synaptic_p_weights_fn_EE, 'w')
        print("{}\t{}".format(
            "time(ms)", "EE(nS)"),
            file=self.weights_fh_EE, flush=True)

        self.synaptic_p_weights_fn_EI = (
            "01-synaptic-weights-EI-" + str(self.rank) + ".txt")
        self.weights_fh_EI = open(
            self.synaptic_p_weights_fn_EI, 'w')
        print("{}\t{}".format(
            "time(ms)", "EI(nS)"),
            file=self.weights_fh_EI, flush=True)

        self.synaptic_p_weights_fn_II = (
            "01-synaptic-weights-II-" + str(self.rank) + ".txt")
        self.weights_fh_II = open(
            self.synaptic_p_weights_fn_II, 'w')
        print("{}\t{}".format(
            "time(ms)", "II(nS)"),
            file=self.weights_fh_II, flush=True)

        self.synaptic_p_weights_fn_IE = (
            "01-synaptic-weights-IE-" + str(self.rank) + ".txt")
        self.weights_fh_IE = open(
            self.synaptic_p_weights_fn_IE, 'w')
        print("{}\t{}".format(
            "time(ms)", "IE(nS)"),
            file=self.weights_fh_IE, flush=True)

        if self.is_str_p_enabled:
            if self.rank == 0:
                self.syn_del_fn_lpz_c_E = (
                    "04-synapses-deleted-lpz_c_E-" + str(self.rank) + ".txt")
                self.syn_del_fh_lpz_c_E = open(
                    self.syn_del_fn_lpz_c_E, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_lpz_c_E, flush=True)

                self.syn_new_fn_lpz_c_E = (
                    "04-synapses-formed-lpz_c_E-" + str(self.rank) + ".txt")
                self.syn_new_fh_lpz_c_E = open(
                    self.syn_new_fn_lpz_c_E, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_lpz_c_E, flush=True)

                self.syn_del_fn_lpz_b_E = (
                    "04-synapses-deleted-lpz_b_E-" + str(self.rank) + ".txt")
                self.syn_del_fh_lpz_b_E = open(
                    self.syn_del_fn_lpz_b_E, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_lpz_b_E, flush=True)

                self.syn_new_fn_lpz_b_E = (
                    "04-synapses-formed-lpz_b_E-" + str(self.rank) + ".txt")
                self.syn_new_fh_lpz_b_E = open(
                    self.syn_new_fn_lpz_b_E, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_lpz_b_E, flush=True)

                self.syn_del_fn_p_lpz_E = (
                    "04-synapses-deleted-p_lpz_E-" + str(self.rank) + ".txt")
                self.syn_del_fh_p_lpz_E = open(
                    self.syn_del_fn_p_lpz_E, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_p_lpz_E, flush=True)

                self.syn_new_fn_p_lpz_E = (
                    "04-synapses-formed-p_lpz_E-" + str(self.rank) + ".txt")
                self.syn_new_fh_p_lpz_E = open(
                    self.syn_new_fn_p_lpz_E, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_p_lpz_E, flush=True)

                self.syn_del_fn_o_E = (
                    "04-synapses-deleted-o_E-" + str(self.rank) + ".txt")
                self.syn_del_fh_o_E = open(
                    self.syn_del_fn_o_E, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_o_E, flush=True)

                self.syn_new_fn_o_E = (
                    "04-synapses-formed-o_E-" + str(self.rank) + ".txt")
                self.syn_new_fh_o_E = open(
                    self.syn_new_fn_o_E, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_o_E, flush=True)

                # inhibitory neurons
                self.syn_del_fn_lpz_c_I = (
                    "04-synapses-deleted-lpz_c_I-" + str(self.rank) + ".txt")
                self.syn_del_fh_lpz_c_I = open(
                    self.syn_del_fn_lpz_c_I, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_lpz_c_I, flush=True)

                self.syn_new_fn_lpz_c_I = (
                    "04-synapses-formed-lpz_c_I-" + str(self.rank) + ".txt")
                self.syn_new_fh_lpz_c_I = open(
                    self.syn_new_fn_lpz_c_I, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_lpz_c_I, flush=True)

                self.syn_del_fn_lpz_b_I = (
                    "04-synapses-deleted-lpz_b_I-" + str(self.rank) + ".txt")
                self.syn_del_fh_lpz_b_I = open(
                    self.syn_del_fn_lpz_b_I, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_lpz_b_I, flush=True)

                self.syn_new_fn_lpz_b_I = (
                    "04-synapses-formed-lpz_b_I-" + str(self.rank) + ".txt")
                self.syn_new_fh_lpz_b_I = open(
                    self.syn_new_fn_lpz_b_I, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_lpz_b_I, flush=True)

                self.syn_del_fn_p_lpz_I = (
                    "04-synapses-deleted-p_lpz_I-" + str(self.rank) + ".txt")
                self.syn_del_fh_p_lpz_I = open(
                    self.syn_del_fn_p_lpz_I, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_p_lpz_I, flush=True)

                self.syn_new_fn_p_lpz_I = (
                    "04-synapses-formed-p_lpz_I-" + str(self.rank) + ".txt")
                self.syn_new_fh_p_lpz_I = open(
                    self.syn_new_fn_p_lpz_I, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_p_lpz_I, flush=True)

                self.syn_del_fn_o_I = (
                    "04-synapses-deleted-o_I-" + str(self.rank) + ".txt")
                self.syn_del_fh_o_I = open(
                    self.syn_del_fn_o_I, 'w')
                print("{}\t{}\t{}\t{}".format(
                    "time(ms)", "gid", "total_conns", "conns_deleted"),
                    file=self.syn_del_fh_o_I, flush=True)

                self.syn_new_fn_o_I = (
                    "04-synapses-formed-o_I-" + str(self.rank) + ".txt")
                self.syn_new_fh_o_I = open(
                    self.syn_new_fn_o_I, 'w')
                print("{}\t{}\t{}".format(
                    "time(ms)", "gid", "conns_gained"),
                    file=self.syn_new_fh_o_I, flush=True)

    def setup_plasticity(self, structural_p=True, synaptic_p=True):
        """Control plasticities."""
        self.is_str_p_enabled = structural_p
        self.is_syn_p_enabled = synaptic_p

        if self.is_str_p_enabled and self.is_syn_p_enabled:
            logging.info("NETWORK SETUP TO HANDLE BOTH PLASTICITIES")
        elif self.is_str_p_enabled and not self.is_syn_p_enabled:
            logging.info("NETWORK SETUP TO HANDLE ONLY STRUCTURAL PLASTICITY")
        elif self.is_syn_p_enabled and not self.is_str_p_enabled:
            logging.info("NETWORK SETUP TO HANDLE ONLY SYNAPTIC PLASTICITY")
        else:
            logging.critical("Both plasticities cannot be disabled. Exiting.")
            sys.exit()

    def prerun_setup(self,
                     stabilisation_time=None,
                     sp_update_interval=None,
                     recording_interval=None):
        """Pre reun configuration."""
        # Cannot be changed mid simulation
        self.update_time_windows(stabilisation_time, sp_update_interval,
                                 recording_interval)
        self.__setup_simulation()

    def print_simulation_parameters(self):
        """Print the parameters of the simulation to a file."""
        if self.rank == 0:
            with open("99-simulation_params.txt", 'w') as pfile:
                print("{}: {} milli seconds".format("dt", self.dt),
                      file=pfile)
                print("{}: {} seconds".format("stabilisation_time",
                                              self.default_stabilisation_time),
                      file=pfile)
                print("{}: {} seconds".format("recording_interval",
                                              self.recording_interval),
                      file=pfile)
                print("{}: {}".format("str_p_enabled",
                                      self.is_str_p_enabled),
                      file=pfile)
                print("{}: {}".format("syn_p_enabled",
                                      self.is_syn_p_enabled),
                      file=pfile)
                print("{}: {}".format("is_rewiring_enabled",
                                      self.is_rewiring_enabled),
                      file=pfile)
                print("{}: {}".format("syn_del_strategy",
                                      self.syn_del_strategy),
                      file=pfile)
                print("{}: {}".format("syn_form_strategy",
                                      self.syn_form_strategy),
                      file=pfile)
                print("{}: {}".format("num_E", self.populations['E']),
                      file=pfile)
                print("{}: {}".format("num_I", self.populations['I']),
                      file=pfile)
                print("{}: {}".format("num_P", self.populations['P']),
                      file=pfile)
                print("{}: {}".format("num_R", self.populations['R']),
                      file=pfile)
                print("{}: {}".format("pattern_percent", self.pattern_percent),
                      file=pfile)
                print("{}: {}".format("recall_percent", self.recall_percent),
                      file=pfile)
                print("{}: {}".format("num_colsE", self.colsE),
                      file=pfile)
                print("{}: {}".format("num_colsI", self.colsI),
                      file=pfile)
                print("{}: {}".format("dist_neuronsE", self.neuronal_distE),
                      file=pfile)
                print("{}: {}".format("dist_neuronsI", self.neuronal_distI),
                      file=pfile)
                print("{}: {}".format("lpz_percent", self.lpz_percent),
                      file=pfile)
                print("{}: {} micro metres".format(
                    "grid_size_E",
                    self.location_tree.data[len(self.neuronsE) - 1]),
                    file=pfile)
                print("{}: {} micro metres".format("sd_dist",
                                                   self.location_sd),
                      file=pfile)
                print("{}: {} seconds".format("sp_update_interval",
                                              self.sp_update_interval),
                      file=pfile)
                print("{}: {} seconds".format("recording_interval",
                                              self.recording_interval),
                      file=pfile)
                print("{}: {} nS".format("wbar", self.wbar),
                      file=pfile)
                print("{}: {} nS".format("weightEE", self.weightEE),
                      file=pfile)
                print("{}: {} ns".format("weightPatternEE",
                                         self.weightPatternEE),
                      file=pfile)
                print("{}: {} nS".format("weightEI", self.weightEI),
                      file=pfile)
                print("{}: {} nS".format("weightII", self.weightII),
                      file=pfile)
                print("{}: {} nS".format("weightExt", self.weightExt),
                      file=pfile)
                print("{}: {}".format("sparsity", self.sparsity),
                      file=pfile)

    def update_time_windows(self,
                            stabilisation_time=None,
                            sp_update_interval=None,
                            recording_interval=None):
        """Set up stabilisation time."""
        if stabilisation_time:
            self.default_stabilisation_time = stabilisation_time
        if sp_update_interval:
            self.sp_update_interval = sp_update_interval
        if recording_interval:
            self.recording_interval = recording_interval

        if math.gcd(self.recording_interval, self.sp_update_interval) == 1:
            logging.warning(
                "Recording interval({}) and SP interval({}) are not multiples")
            logging.warning(
                "Simulation will run in 1 second chunks only")

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
        self.__get_regions()
        self.__setup_detectors()
        self.__setup_initial_connection_params()
        self.__create_initial_connections()
        self.__setup_files()

        self.dump_data()

    def stabilise(self, stab_time=0., label="Stabilising"):
        """Stabilise network."""
        # use default if not mentioned
        if stab_time:
            stabilisation_time = stab_time
        else:
            stabilisation_time = self.default_stabilisation_time

        logging.debug("SIMULATION: STABILISING for {} seconds".format(
            stabilisation_time))
        self.run_sim_phase(stabilisation_time, label)

    def run_sim_phase(self, sim_time=2000, label="Phase A"):
        """Run a simulation phase."""
        # take the smaller of the two intervals
        current_sim_time = nest.GetKernelStatus()['time']
        logging.info("Phase started at {}: {}".format(
            current_sim_time, label))
        phase_time = 0

        # make sure we run for the smallest interval
        if self.is_rewiring_enabled:
            run_duration = math.gcd(self.sp_update_interval,
                                    self.recording_interval)
        else:
            run_duration = self.recording_interval

        if sim_time < run_duration:
            logging.warning(
                "Requested run time ({}) < minimum run duration ({})".format(
                    sim_time, run_duration))
            logging.warning(
                "Setting run time to run duration")
            sim_time = run_duration

        update_steps = numpy.arange(0, sim_time, run_duration)
        for i in update_steps:
            nest.Simulate(run_duration*1000)
            current_sim_time = nest.GetKernelStatus()['time']
            logging.info("Simulation time: {} seconds".format(
                current_sim_time/1000))
            # so it's always a multiple of the smallest simulation run time
            phase_time += run_duration

            # dump data
            if int(phase_time % self.recording_interval) == 0:
                self.dump_data()
            # update connectivity
            if int(phase_time % self.sp_update_interval) == 0:
                self.update_connectivity()

        current_sim_time = nest.GetKernelStatus()['time']
        logging.info("Phase ended at {}: {}".format(
            current_sim_time, label))

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
        logging.debug("Got {} local neurons on rank {}".format(
            len(local_neurons), self.rank))

        lneurons = nest.GetStatus(local_neurons, ['global_id',
                                                  'synaptic_elements'])
        # returns a list of sets - one set from each rank
        ranksets = self.comm.allgather(lneurons)
        logging.debug("Got {} ranksets".format(len(ranksets)))

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
                # total elms cannot be less than 0
                # connected elms cannot be less than 0
                # continuous = False, so all values are already ints here
                delta_z_ax = int(source_elms_total - source_elms_con)
                delta_z_d_ex = int(target_elms_total_ex - target_elms_con_ex)
                delta_z_d_in = int(target_elms_total_in - target_elms_con_in)

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

        logging.debug(
            "Got synaptic elements for {} neurons.".format(
                len(synaptic_elms)))
        return synaptic_elms

    def __get_del_ps_w(self, anchor, options, num_required):
        """Choose partners to delete based on weight of connections."""
        optdict = {}
        for opt in options:
            optdict[opt[0]] = opt[1]

        sortedoptions = dict(sorted(optdict.items(),
                                    key=operator.itemgetter(1)))
        weakestoptions = list(sortedoptions.keys())[0:num_required:]
        logging.debug("Returning weakest links")
        return weakestoptions

    def __get_del_ps_d(self, anchor, options, num_required):
        """Choose partners to delete based on distance."""
        distances = {}
        anchorloc = numpy.array(
            self.location_tree.data[anchor - self.neuronsE[0]])
        for opt in options:
            location = numpy.array(
                self.location_tree.data[opt - self.neuronsE[0]])
            distance = numpy.linalg.norm(location - anchorloc)
            distances[opt] = distance

        sorted_distances = dict(sorted(distances.items(),
                                       key=operator.itemgetter(1)))
        farthest_opts = list(sorted_distances.keys())[-num_required:]
        logging.debug("Returning closest partners")

        return farthest_opts

    def __delete_connections_from_pre(self, synelms):
        """Delete connections when the neuron is a source."""
        logging.debug
        ("Deleting connections from pre using '{}' deletion strategy".format(
            self.syn_del_strategy))
        total_synapses = 0
        syn_del_lpz_c_E = 0
        syn_del_lpz_b_E = 0
        syn_del_p_lpz_E = 0
        syn_del_o_E = 0
        syn_del_lpz_c_I = 0
        syn_del_lpz_b_I = 0
        syn_del_p_lpz_I = 0
        syn_del_o_I = 0
        current_sim_time = (str(nest.GetKernelStatus()['time']))
        # the order in which these are removed should not matter - whether we
        # remove connections using axons first or dendrites first, the end
        # state of the network should be the same.
        # Note that we are modifying a dictionary while iterating over it. This
        # is OK here since we're not modifying the keys, only the values.
        # http://stackoverflow.com/a/2315529/375067
        for nrn in (self.neuronsE + self.neuronsI):
            gid = nrn
            elms = synelms[nrn]
            partner = 0  # for exception
            total_synapses_this_gid = 0
            syn_del_this_gid = 0
            try:
                # excitatory neurons as sources
                if 'Axon_ex' in elms and elms['Axon_ex'] < 0:
                    # GetConnections only returns connections who have targets
                    # on this particular rank. So I need to allgather to
                    # collect all targets.
                    chosen_targets = []
                    conns = []
                    conns = nest.GetConnections(
                        source=[gid], synapse_model='static_synapse_ex')
                    localtargets = []
                    if self.syn_del_strategy == "weight":
                        # also need to store weight
                        # list of lists: [[target, weight], [target, weight]..]
                        weights = nest.GetStatus(conns, "weight")
                        for i in range(0, len(conns)):
                            localtargets.append(
                                [conns[i][1], weights[i]])
                    else:
                        for acon in conns:
                            localtargets.append(acon[1])

                    alltargets = self.comm.allgather(localtargets)
                    targets = [t for sublist in alltargets for t in sublist]
                    total_synapses_this_gid = len(targets)
                    if len(targets) > 0:
                        # this is where the selection logic is
                        if len(targets) > int(abs(elms['Axon_ex'])):
                            if self.syn_del_strategy == "random":
                                # Doesn't merit a new method
                                chosen_targets = random.sample(
                                    targets, int(abs(elms['Axon_ex'])))
                            elif self.syn_del_strategy == "distance":
                                chosen_targets = self.__get_del_ps_d(
                                    gid, targets, int(abs(elms['Axon_ex'])))
                            elif self.syn_del_strategy == "weight":
                                chosen_targets = self.__get_del_ps_w(
                                    gid, targets, int(abs(elms['Axon_ex'])))
                        else:
                            chosen_targets = targets
                        logging.debug(
                            "Rank {}: {}/{} tgts chosen for neuron {}".format(
                                self.rank, len(chosen_targets),
                                total_synapses_this_gid, gid))

                        for t in chosen_targets:
                            syn_del_this_gid += 1
                            partner = t
                            nest.Disconnect(
                                pre=[gid], post=[t], syn_spec={
                                    'model': 'static_synapse_ex',
                                    'pre_synaptic_element': 'Axon_ex',
                                    'post_synaptic_element': 'Den_ex',
                                }, conn_spec={
                                    'rule': 'one_to_one'}
                            )
                            synelms[t]['Den_ex'] += 1

                # inhibitory neurons as sources
                # here, there can be two types of targets, E neurons or
                # I neurons, and they must each be treated separately
                elif 'Axon_in' in elms and elms['Axon_in'] < 0:
                    connsToI = nest.GetConnections(
                        source=[gid], synapse_model='static_synapse_in')
                    connsToE = nest.GetConnections(
                        source=[gid], synapse_model='stdp_synapse_in')

                    localtargetsI = []
                    localtargetsE = []
                    chosen_targets = []

                    if self.syn_del_strategy == "weight":
                        # also need to store weight
                        # list of lists: [[target, weight], [target, weight]..]
                        weightsToI = nest.GetStatus(connsToI, "weight")
                        weightsToE = nest.GetStatus(connsToE, "weight")
                        for i in range(0, len(connsToI)):
                            localtargetsI.append(
                                [connsToI[i][1], weightsToI[i]])
                        for i in range(0, len(connsToE)):
                            localtargetsE.append(
                                [connsToE[i][1], weightsToE[i]])
                    else:
                        for acon in connsToI:
                            localtargetsI.append(acon[1])
                        for acon in connsToE:
                            localtargetsE.append(acon[1])

                    alltargetsI = self.comm.allgather(localtargetsI)
                    alltargetsE = self.comm.allgather(localtargetsE)

                    targetsI = [t for sublist in alltargetsI for t in sublist]
                    targetsE = [t for sublist in alltargetsE for t in sublist]

                    total_synapses_this_gid = (len(targetsE) + len(targetsI))

                    if (total_synapses_this_gid) > 0:
                        # this is where the selection logic is
                        if (total_synapses_this_gid) > \
                                int(abs(elms['Axon_in'])):
                            if self.syn_del_strategy == "random":
                                # Doesn't merit a new method
                                chosen_targets = random.sample(
                                    (targetsE + targetsI),
                                    int(abs(elms['Axon_in'])))
                            elif self.syn_del_strategy == "distance":
                                chosen_targets = self.__get_del_ps_d(
                                    gid, (targetsE + targetsI),
                                    int(abs(elms['Axon_in'])))
                            elif self.syn_del_strategy == "weight":
                                chosen_targets = self.__get_del_ps_w(
                                    gid, (targetsE + targetsI),
                                    int(abs(elms['Axon_in'])))
                        else:
                            chosen_targets = (targetsE + targetsI)
                        logging.debug(
                            "Rank {}: {}/{} tgts chosen for neuron {}".format(
                                self.rank, len(chosen_targets),
                                total_synapses_this_gid, gid))

                        for t in chosen_targets:
                            synelms[t]['Den_in'] += 1
                            syn_del_this_gid += 1
                            partner = t
                            if t in targetsE:
                                nest.Disconnect(
                                    pre=[gid], post=[t], syn_spec={
                                        'model': 'stdp_synapse_in',
                                        'pre_synaptic_element': 'Axon_in',
                                        'post_synaptic_element': 'Den_in',
                                    }, conn_spec={
                                        'rule': 'one_to_one'}
                                )
                            else:
                                nest.Disconnect(
                                    pre=[gid], post=[t], syn_spec={
                                        'model': 'static_synapse_in',
                                        'pre_synaptic_element': 'Axon_in',
                                        'post_synaptic_element': 'Den_in',
                                    }, conn_spec={
                                        'rule': 'one_to_one'}
                                )

                total_synapses += total_synapses_this_gid
                if self.rank == 0:
                    if syn_del_this_gid > 0:
                        if gid in self.lpz_c_neurons_E:
                            fh = self.syn_del_fh_lpz_c_E
                            syn_del_lpz_c_E += syn_del_this_gid
                        elif gid in self.lpz_b_neurons_E:
                            fh = self.syn_del_fh_lpz_b_E
                            syn_del_lpz_b_E += syn_del_this_gid
                        elif gid in self.p_lpz_neurons_E:
                            fh = self.syn_del_fh_p_lpz_E
                            syn_del_p_lpz_E += syn_del_this_gid
                        elif gid in self.o_neurons_E:
                            fh = self.syn_del_fh_o_E
                            syn_del_o_E += syn_del_this_gid
                        elif gid in self.lpz_c_neurons_I:
                            fh = self.syn_del_fh_lpz_c_I
                            syn_del_lpz_c_I += syn_del_this_gid
                        elif gid in self.lpz_b_neurons_I:
                            fh = self.syn_del_fh_lpz_b_I
                            syn_del_lpz_b_I += syn_del_this_gid
                        elif gid in self.p_lpz_neurons_I:
                            fh = self.syn_del_fh_p_lpz_I
                            syn_del_p_lpz_I += syn_del_this_gid
                        elif gid in self.o_neurons_I:
                            fh = self.syn_del_fh_o_I
                            syn_del_o_I += syn_del_this_gid

                        print("{}\t{}\t{}\t{}".format(
                            current_sim_time, gid, total_synapses_this_gid,
                            syn_del_this_gid),
                            file=fh)

            except KeyError as e:
                logging.critical("KeyError exception while disconnecting!")
                logging.critical("GID: {} : {}".format(gid, synelms[gid]))
                logging.critical(
                    "Partner id: {} : {}".format(
                        partner, synelms[partner]))
                logging.critical("Exception: {}".format(str(e)))
                raise
            except:
                logging.critical("Some other exception")
                raise

        logging.debug(
            "{} of {} connections deleted from pre".format(
                (syn_del_lpz_c_E + syn_del_lpz_b_E +
                 syn_del_p_lpz_E + syn_del_lpz_c_I +
                 syn_del_lpz_b_I + syn_del_p_lpz_I +
                 syn_del_o_E + syn_del_o_I),
                total_synapses))

    def __delete_connections_from_post(self, synelms):
        """Delete connections when neuron is target."""
        logging.debug(
            "Deleting conns from post using '{}' deletion strategy".format(
                self.syn_del_strategy))
        total_synapses = 0
        syn_del_lpz_c_E = 0
        syn_del_lpz_b_E = 0
        syn_del_p_lpz_E = 0
        syn_del_o_E = 0
        syn_del_lpz_c_I = 0
        syn_del_lpz_b_I = 0
        syn_del_p_lpz_I = 0
        syn_del_o_I = 0
        current_sim_time = (str(nest.GetKernelStatus()['time']))
        # excitatory dendrites as targets
        # weight dependent deletion doesn't apply - all synapses have
        # same weight
        for nrn in (self.neuronsE + self.neuronsI):
            gid = nrn
            elms = synelms[nrn]
            partner = 0  # for exception
            total_synapses_this_gid = 0
            syn_del_this_gid = 0
            try:
                if 'Den_ex' in elms and elms['Den_ex'] < 0:
                    conns = nest.GetConnections(
                        target=[gid], synapse_model='static_synapse_ex')
                    localsources = []
                    chosen_sources = []
                    if self.syn_del_strategy == "weight":
                        # also need to store weight
                        # list of lists: [[target, weight], [target, weight]..]
                        weights = nest.GetStatus(conns, "weight")
                        for i in range(0, len(conns)):
                            localsources.append(
                                [conns[i][0], weights[i]])
                    else:
                        for acon in conns:
                            localsources.append(acon[0])

                    allsources = self.comm.allgather(localsources)
                    sources = [s for sublist in allsources for s in sublist]
                    total_synapses_this_gid += len(sources)

                    if len(sources) > 0:
                        if len(sources) > int(abs(elms['Den_ex'])):
                            if self.syn_del_strategy == "random":
                                chosen_sources = random.sample(
                                    sources, int(abs(elms['Den_ex'])))
                            elif self.syn_del_strategy == "distance":
                                chosen_sources = self.__get_del_ps_d(
                                    gid, sources, int(abs(elms['Den_ex'])))
                            elif self.syn_del_strategy == "weight":
                                chosen_sources = self.__get_del_ps_w(
                                    gid, sources, int(abs(elms['Den_ex'])))
                        else:
                            chosen_sources = sources
                        logging.debug(
                            "Rank {}: {}/{} srcs chosen for neuron {}".format(
                                self.rank, len(chosen_sources),
                                total_synapses_this_gid, gid))

                        for s in chosen_sources:
                            syn_del_this_gid += 1
                            partner = s
                            nest.Disconnect(
                                pre=[s], post=[gid], syn_spec={
                                    'model': 'static_synapse_ex',
                                    'pre_synaptic_element': 'Axon_ex',
                                    'post_synaptic_element': 'Den_ex',
                                }, conn_spec={
                                    'rule': 'one_to_one'}
                            )
                            synelms[s]['Axon_ex'] += 1

                # inhibitory dendrites as targets
                if 'Den_in' in elms and elms['Den_in'] < 0:
                    # is it an inhibitory neuron?
                    if 'Axon_in' in elms:
                        # all synapses are of same weight, so weight dependent
                        # not required
                        conns = nest.GetConnections(
                            target=[gid], synapse_model='static_synapse_in')
                        localsources = []
                        chosen_sources = []
                        for acon in conns:
                            localsources.append(acon[0])

                        allsources = self.comm.allgather(localsources)
                        sources = [s for sublist in allsources for s in
                                   sublist]
                        total_synapses_this_gid += len(sources)

                        if len(sources) > 0:
                            if len(sources) > int(abs(elms['Den_in'])):
                                if self.syn_del_strategy == "random" or \
                                        self.syn_del_strategy == "weight":
                                    chosen_sources = random.sample(
                                        sources, int(abs(elms['Den_in'])))
                                elif self.syn_del_strategy == "distance":
                                    chosen_sources = self.__get_del_ps_d(
                                        gid, sources, int(abs(elms['Den_in'])))
                            else:
                                chosen_sources = sources
                            logging.debug(
                                "Rank {}: {}/{} srcs chosen for nrn {}".format(
                                    self.rank, len(chosen_sources),
                                    total_synapses_this_gid, gid))

                            for s in chosen_sources:
                                syn_del_this_gid += 1
                                partner = s
                                nest.Disconnect(
                                    pre=[s], post=[gid], syn_spec={
                                        'model': 'static_synapse_in',
                                        'pre_synaptic_element': 'Axon_in',
                                        'post_synaptic_element': 'Den_in',
                                    }, conn_spec={
                                        'rule': 'one_to_one'}
                                )
                                synelms[s]['Axon_in'] += 1

                    # it's an excitatory neuron
                    else:
                        conns = nest.GetConnections(
                            target=[gid], synapse_model='stdp_synapse_in')
                        localsources = []
                        chosen_sources = []
                        if self.syn_del_strategy == "weight":
                            # also need to store weight
                            # list of lists: [[source, weight], [source,
                            # weight]..]
                            weights = nest.GetStatus(conns, "weight")
                            for i in range(0, len(conns)):
                                localsources.append([conns[i][0], weights[i]])
                        else:
                            # otherwise only sids
                            for acon in conns:
                                localsources.append(acon[0])
                        allsources = self.comm.allgather(localsources)
                        sources = [s for sublist in allsources for s in
                                   sublist]
                        total_synapses_this_gid += len(sources)

                        if len(sources) > 0:
                            if len(sources) > int(abs(elms['Den_in'])):
                                if self.syn_del_strategy == "random":
                                    chosen_sources = random.sample(
                                        sources, int(abs(elms['Den_in'])))
                                elif self.syn_del_strategy == "distance":
                                    chosen_sources = self.__get_del_ps_d(
                                        gid, sources, int(abs(elms['Den_in'])))
                                elif self.syn_del_strategy == "weight":
                                    chosen_sources = self.__get_del_ps_w(
                                        gid, sources, int(abs(elms['Den_in'])))
                            else:
                                chosen_sources = sources
                            logging.debug(
                                "Rank {}: {}/{} srcs chosen for nrn {}".format(
                                    self.rank, len(chosen_sources),
                                    total_synapses_this_gid, gid))

                            for s in chosen_sources:
                                syn_del_this_gid += 1
                                partner = s
                                nest.Disconnect(
                                    pre=[s], post=[gid], syn_spec={
                                        'model': 'stdp_synapse_in',
                                        'pre_synaptic_element': 'Axon_in',
                                        'post_synaptic_element': 'Den_in',
                                    }, conn_spec={
                                        'rule': 'one_to_one'}
                                )
                                synelms[s]['Axon_in'] += 1

                total_synapses += total_synapses_this_gid
                if self.rank == 0:
                    if syn_del_this_gid > 0:
                        if gid in self.lpz_c_neurons_E:
                            fh = self.syn_del_fh_lpz_c_E
                            syn_del_lpz_c_E += syn_del_this_gid
                        elif gid in self.lpz_b_neurons_E:
                            fh = self.syn_del_fh_lpz_b_E
                            syn_del_lpz_b_E += syn_del_this_gid
                        elif gid in self.p_lpz_neurons_E:
                            fh = self.syn_del_fh_p_lpz_E
                            syn_del_p_lpz_E += syn_del_this_gid
                        elif gid in self.o_neurons_E:
                            fh = self.syn_del_fh_o_E
                            syn_del_o_E += syn_del_this_gid
                        elif gid in self.lpz_c_neurons_I:
                            fh = self.syn_del_fh_lpz_c_I
                            syn_del_lpz_c_I += syn_del_this_gid
                        elif gid in self.lpz_b_neurons_I:
                            fh = self.syn_del_fh_lpz_b_I
                            syn_del_lpz_b_I += syn_del_this_gid
                        elif gid in self.p_lpz_neurons_I:
                            fh = self.syn_del_fh_p_lpz_I
                            syn_del_p_lpz_I += syn_del_this_gid
                        elif gid in self.o_neurons_I:
                            fh = self.syn_del_fh_o_I
                            syn_del_o_I += syn_del_this_gid

                        print("{}\t{}\t{}\t{}".format(
                            current_sim_time, gid, total_synapses_this_gid,
                            syn_del_this_gid),
                            file=fh)

            except KeyError as e:
                logging.critical("KeyError exception while disconnecting!")
                logging.critical("GID: {} : {}".format(gid, synelms[gid]))
                logging.critical(
                    "Partner id: {} : {}".format(
                        partner, synelms[partner]))
                logging.critical("Exception: {}".format(str(e)))
                raise
            except:
                logging.critical("Some other exception")
                raise

        logging.debug(
            "{} of {} connections deleted from post".format(
                (syn_del_lpz_c_E + syn_del_lpz_b_E +
                 syn_del_p_lpz_E + syn_del_lpz_c_I +
                 syn_del_lpz_b_I + syn_del_p_lpz_I +
                 syn_del_o_E + syn_del_o_I),
                total_synapses))

    def __get_form_part_d(self, source, options, num_required):
        """Choose partners based on distance."""
        # in efficient, but works.
        distances = {}
        sourceloc = numpy.array(self.location_tree.data[source -
                                                        self.neuronsE[0]])
        for opt in options:
            location = numpy.array(self.location_tree.data[opt -
                                                           self.neuronsE[0]])
            distance = numpy.linalg.norm(location - sourceloc)
            distances[opt] = distance

        sorted_distances = dict(sorted(distances.items(),
                                       key=operator.itemgetter(1)))
        nearest_opts = list(sorted_distances.keys())[0:num_required]
        logging.debug("Returning closest partners")

        return nearest_opts

    def __create_connections(self, synelms):
        """Connect random neurons to create new connections."""
        logging.debug("Creating connections using the {} strategy".format(
            self.syn_form_strategy))
        syn_new_lpz_c_E = 0
        syn_new_lpz_b_E = 0
        syn_new_p_lpz_E = 0
        syn_new_o_E = 0
        syn_new_lpz_c_I = 0
        syn_new_lpz_b_I = 0
        syn_new_p_lpz_I = 0
        syn_new_o_I = 0
        current_sim_time = (str(nest.GetKernelStatus()['time']))
        for nrn in (self.neuronsE + self.neuronsI):
            syn_new_this_gid = 0
            total_options_this_gid = 0
            gid = nrn
            elms = synelms[nrn]
            chosen_targets = []
            # excitatory connections - only need to look at Axons, it doesn't
            # matter which synaptic elements you start with, whichever are less
            # will act as the limiting factor.
            if 'Axon_ex' in elms and elms['Axon_ex'] > 0:
                targetsE = []
                targetsI = []

                for atarget in (self.neuronsE + self.neuronsI):
                    tid = atarget
                    telms = synelms[atarget]
                    if 'Den_ex' in telms and telms['Den_ex'] > 0:
                        # add the target multiple times, since it has multiple
                        # available contact points
                        if 'Axon_ex' in telms:
                            targetsE.extend([tid]*int(telms['Den_ex']))
                        else:
                            targetsI.extend([tid]*int(telms['Den_ex']))

                total_options_this_gid = len(targetsE) + len(targetsI)
                if (total_options_this_gid) > 0:
                    if (total_options_this_gid) > int(abs(elms['Axon_ex'])):
                        if self.syn_form_strategy == "random":
                            chosen_targets = random.sample(
                                (targetsE + targetsI),
                                int(abs(elms['Axon_ex'])))
                        elif self.syn_form_strategy == "distance":
                            chosen_targets = self.__get_form_part_d(
                                gid, (targetsE + targetsI),
                                int(abs(elms['Axon_ex'])))
                    else:
                        chosen_targets = (targetsE + targetsI)
                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_targets),
                            total_options_this_gid, gid))

                    for cho in chosen_targets:
                        synelms[cho]['Den_ex'] -= 1
                        syn_new_this_gid += 1
                        if cho in targetsE:
                            nest.Connect([gid], [cho],
                                         conn_spec='one_to_one',
                                         syn_spec=self.synDictEE)
                        else:
                            nest.Connect([gid], [cho],
                                         conn_spec='one_to_one',
                                         syn_spec=self.synDictEI)

            # here, you can connect either with E neurons or I neurons but both
            # will have different synapse types. So, a bit more work required
            # here than with the Axon_ex which always forms the same type of
            # synapse
            elif 'Axon_in' in elms and elms['Axon_in'] > 0:
                targetsE = []
                targetsI = []

                for atarget in (self.neuronsE + self.neuronsI):
                    tid = atarget
                    telms = synelms[atarget]
                    if 'Den_in' in telms and telms['Den_in'] > 0:
                        # add the target multiple times, since it has multiple
                        # available contact points
                        if 'Axon_ex' in telms:
                            targetsE.extend([tid]*int(telms['Den_in']))
                        else:
                            targetsI.extend([tid]*int(telms['Den_in']))

                total_options_this_gid = len(targetsE) + len(targetsI)
                if (total_options_this_gid) > 0:
                    if (total_options_this_gid) > int(abs(elms['Axon_in'])):
                        if self.syn_form_strategy == "random":
                            chosen_targets = random.sample(
                                (targetsE + targetsI),
                                int(abs(elms['Axon_in'])))
                        elif self.syn_form_strategy == "distance":
                            chosen_targets = self.__get_form_part_d(
                                gid, (targetsE + targetsI),
                                int(abs(elms['Axon_in'])))
                    else:
                        chosen_targets = (targetsE + targetsI)
                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_targets),
                            total_options_this_gid, gid))

                    for target in chosen_targets:
                        syn_new_this_gid += 1
                        synelms[target]['Den_in'] -= 1
                        if target in targetsE:
                            nest.Connect([gid], [target],
                                         conn_spec='one_to_one',
                                         syn_spec=self.synDictIE)
                        else:
                            nest.Connect([gid], [target],
                                         conn_spec='one_to_one',
                                         syn_spec=self.synDictII)

            if self.rank == 0:
                if syn_new_this_gid > 0:
                    if gid in self.lpz_c_neurons_E:
                        fh = self.syn_new_fh_lpz_c_E
                        syn_new_lpz_c_E += syn_new_this_gid
                    elif gid in self.lpz_b_neurons_E:
                        fh = self.syn_new_fh_lpz_b_E
                        syn_new_lpz_b_E += syn_new_this_gid
                    elif gid in self.p_lpz_neurons_E:
                        fh = self.syn_new_fh_p_lpz_E
                        syn_new_p_lpz_E += syn_new_this_gid
                    elif gid in self.o_neurons_E:
                        fh = self.syn_new_fh_o_E
                        syn_new_o_E += syn_new_this_gid
                    elif gid in self.lpz_c_neurons_I:
                        fh = self.syn_new_fh_lpz_c_I
                        syn_new_lpz_c_I += syn_new_this_gid
                    elif gid in self.lpz_b_neurons_I:
                        fh = self.syn_new_fh_lpz_b_I
                        syn_new_lpz_b_I += syn_new_this_gid
                    elif gid in self.p_lpz_neurons_I:
                        fh = self.syn_new_fh_p_lpz_I
                        syn_new_p_lpz_I += syn_new_this_gid
                    elif gid in self.o_neurons_I:
                        fh = self.syn_new_fh_o_I
                        syn_new_o_I += syn_new_this_gid

                    print("{}\t{}\t{}".format(
                        current_sim_time, gid,
                        syn_new_this_gid),
                        file=fh)

        if self.rank == 0:
            logging.debug(
                "{} new connections created".format(
                    (syn_new_lpz_c_E + syn_new_lpz_b_E +
                     syn_new_p_lpz_E + syn_new_lpz_c_I +
                     syn_new_lpz_b_I + syn_new_p_lpz_I +
                     syn_new_o_E + syn_new_o_I)))

    def update_connectivity(self):
        """Our implementation of structural plasticity."""
        if not self.is_rewiring_enabled:
            return
        logging.debug("STRUCTURAL PLASTICITY: Updating connectivity")
        syn_elms = self.__get_syn_elms()
        self.__delete_connections_from_pre(syn_elms)
        # Must wait for all ranks to finish before proceeding

        syn_elms_1 = self.__get_syn_elms()
        self.__delete_connections_from_post(syn_elms_1)
        # Must wait for all ranks to finish before proceeding

        syn_elms_2 = self.__get_syn_elms()
        self.__create_connections(syn_elms_2)
        # Must wait for all ranks to finish before proceeding
        logging.debug("STRUCTURAL PLASTICITY: Connectivity updated")

    def invoke_metaplasticity(self):
        """Update growth curve parameters."""
        if self.is_metaplasticity_enabled:
            cal_e = [stat['Ca'] for stat in nest.GetStatus(self.neuronsE) if
                     stat['local']]
            cal_i = [stat['Ca'] for stat in nest.GetStatus(self.neuronsI) if
                     stat['local']]
            self.__set_str_p_hom_point(cal_e, cal_i)
        logging.debug("META PLASTICITY: Growth curves updated")

    def __get_neurons_from_region(self, num_neurons, first_point, last_point):
        """Get neurons in the centre of the grid.

        This will be used to get neurons for deaff, and also to get neurons for
        the centred pattern.
        """
        mid_point = [(x + y)/2 for x, y in zip(last_point, first_point)]
        neurons = self.location_tree.query(
            mid_point, k=num_neurons)[1]
        logging.debug("Got {}/{} neurons".format(len(neurons), num_neurons))
        return neurons

    def __strengthen_pattern_connections(self, pattern_neurons):
        """Strengthen connections that make up the pattern."""
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        nest.SetStatus(connections, {"weight": self.weightPatternEE})
        logging.debug("ANKUR>> Number of connections strengthened: "
                      "{}".format(len(connections)))

    def __track_pattern(self, pattern_neurons):
        """Track the pattern."""
        logging.debug("Tracking this pattern")
        self.patterns.append(pattern_neurons)
        background_neurons = list(
            set(self.neuronsE) - set(pattern_neurons))
        # print to file
        # NOTE: since these are E neurons, the indices match in the location
        # tree. No need to subtract self.neuronsE[0] to get the right indices
        # at the moment. But keep in mind in case something changes in the
        # future.
        if self.rank == 0:
            fn = "00-pattern-neurons-{}.txt".format(
                self.pattern_count)
            with open(fn, 'w') as fh:
                print("gid\txcor\tycor", file=fh, flush=True)
                for neuron in pattern_neurons:
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[neuron - 1][0],
                        self.location_tree.data[neuron - 1][1]),
                        file=fh)

            # background neurons
            fn = "00-background-neurons-{}.txt".format(
                self.pattern_count)

            with open(fn, 'w') as fh:
                print("gid\txcor\tycor", file=fh, flush=True)
                for neuron in background_neurons:
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[neuron - 1][0],
                        self.location_tree.data[neuron - 1][1]),
                        file=fh)

        # set up spike detectors
        sd_params = self.sd_paramsP.copy()
        sd_params['label'] = (sd_params['label'] + "-{}".format(
            self.pattern_count))
        # pattern
        pattern_spike_detector = nest.Create(
            'spike_detector', params=sd_params)
        nest.Connect(pattern_neurons, pattern_spike_detector)
        # save the detector
        self.sdP.append(pattern_spike_detector)

        # background
        sd_params = self.sd_paramsB.copy()
        sd_params['label'] = (sd_params['label'] + "-{}".format(
            self.pattern_count))
        background_spike_detector = nest.Create(
            'spike_detector', params=sd_params)
        nest.Connect(background_neurons, background_spike_detector)
        # save the detector
        self.sdB.append(background_spike_detector)

    def store_pattern_off_centre(self, offset=[0., 0.], track=False):
        """Store a pattern in the centre of LPZ."""
        logging.debug(
            "SIMULATION: Storing pattern {} in centre of LPZ".format(
                self.pattern_count + 1))
        # first E neuron
        first_point = numpy.array(self.location_tree.data[0])
        # last E neuron
        # I neurons are spread among the E neurons and therefore do not make it
        # to the extremeties
        last_point = numpy.array(
            self.location_tree.data[len(self.neuronsE) - 1])
        centre_point = numpy.array(offset) + (first_point + last_point)/2
        self.store_pattern_with_centre(centre_point,
                                       (1.25 * self.populations['P']),
                                       track=True)

    def store_pattern_with_centre(self, centre_point, num_neurons,
                                  track=False):
        """Store a pattern by specifying area extent."""
        logging.debug(
            "SIMULATION: Storing pattern {} centred at:".format(
                self.pattern_count + 1, centre_point))
        self.pattern_count += 1
        # get 1000 neurons - 800 will be E and 200 will be I
        # we only need the 800 I neurons
        all_neurons = self.location_tree.query(
            centre_point, k=num_neurons)[1]
        pattern_neurons = list(set(all_neurons).intersection(
            set(self.neuronsE)))
        self.__strengthen_pattern_connections(pattern_neurons)
        if track:
            self.__track_pattern(pattern_neurons)
        logging.debug(
            "Number of patterns stored: {}".format(
                self.pattern_count))

    def setup_pattern_for_recall(self, pattern_number):
        """
        Set up a pattern for recall.

        Creates a new poisson generator and connects it to a recall subset of
        this pattern - the poisson stimulus will run for the set
        recall_duration from the invocation of this method.
        """
        # set up external stimulus
        pattern_neurons = self.patterns[pattern_number - 1]
        # Only neurons that have are not in the LPZ will be given stimulus
        active_pattern_neurons = list(set(pattern_neurons) -
                                      set(self.lpz_neurons_E))
        # Pick percent of neurons that are not in the LPZ
        recall_neurons = []
        if len(active_pattern_neurons) > 0:
            num_recall_neurons = int(math.ceil(len(active_pattern_neurons) *
                                               self.recall_percent))
            # if the number of active pattern neurons is too small, use the
            # whole lot
            if num_recall_neurons > 0:
                recall_neurons = active_pattern_neurons[-num_recall_neurons:]
            else:
                recall_neurons = active_pattern_neurons

            stim_time = nest.GetKernelStatus()['time']
            neuronDictStim = {'rate': 200.,
                              'origin': stim_time,
                              'start': 0., 'stop': self.recall_duration}
            stim = nest.Create('poisson_generator', 1,
                               neuronDictStim)

            nest.Connect(stim, recall_neurons,
                         conn_spec=self.connDictStim)

            logging.debug("ANKUR>> Number of recall neurons for pattern"
                          "{}: {}".format(pattern_number, len(recall_neurons)))
        else:
            logging.debug("ANKUR>> Pattern {} appears to be completely "
                          "deafferentated - not setting up a recall stimulus")
        self.recall_neurons.append(recall_neurons)

    def recall_last_pattern(self, time):
        """
        Only setup the last pattern.

        An extra helper method, since we'll be doing this most.
        """
        logging.info("SIMULATION: RECALLING LAST PATTERN")
        self.recall_pattern(time, self.pattern_count)

    def recall_pattern(self, time, pattern_number):
        """Recall a pattern."""
        self.setup_pattern_for_recall(pattern_number)
        self.run_sim_phase(
            time, label="Recalling pattern {}".format(pattern_number))

    def deaff_network(self):
        """Deaff a the network."""
        logging.info("SIMULATION: deaffing spatial network")
        for nrn in self.lpz_neurons_E:
            nest.DisconnectOneToOne(self.poissonExt[0], nrn,
                                    syn_spec={'model': 'static_synapse'})
        for nrn in self.lpz_neurons_I:
            nest.DisconnectOneToOne(self.poissonExt[0], nrn,
                                    syn_spec={'model': 'static_synapse'})
        logging.info("SIMULATION: Network deafferentated")

    def __dump_neuron_set(self, fn, neurons):
        """Dump a set of neuronIDs to a text file."""
        with open(fn, 'w') as fh:
            for neuron in neurons:
                print(neuron, file=fh)

    def __dump_ca_concentration(self):
        """Dump calcium concentration."""
        current_sim_time = (str(nest.GetKernelStatus()['time']))
        ca_fn_lpz_c_E = ("02-calcium-lpz_c_E-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_lpz_c_E, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.lpz_c_neurons_E) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

        ca_fn_lpz_b_E = ("02-calcium-lpz_b_E-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_lpz_b_E, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.lpz_b_neurons_E) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

        ca_fn_p_lpz_E = ("02-calcium-p_lpz_E-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_p_lpz_E, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.p_lpz_neurons_E) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)
        # I neurons
        ca_fn_lpz_c_I = ("02-calcium-lpz_c_I-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_lpz_c_I, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.lpz_c_neurons_I) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

        ca_fn_lpz_b_I = ("02-calcium-lpz_b_I-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_lpz_b_I, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.lpz_b_neurons_I) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

        ca_fn_p_lpz_I = ("02-calcium-p_lpz_I-" +
                         str(self.rank) + "-" + current_sim_time +
                         ".txt")
        with open(ca_fn_p_lpz_I, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.p_lpz_neurons_I) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

    def __dump_synaptic_elements_per_neurons(self):
        """
        Dump synaptic elements for each neuron for a time.

        neuronid    ax_total    ax_connected    den_ex_total ...
        """
        if self.is_str_p_enabled:
            current_sim_time = (str(nest.GetKernelStatus()['time']))
            se_fn_lpz_c_E = (
                "05-se-lpz_c_E-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_lpz_c_E, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.lpz_c_neurons_E) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_ex']['z']
                    axons_conn = syn_elms['Axon_ex']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

            se_fn_lpz_b_E = (
                "05-se-lpz_b_E-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_lpz_b_E, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.lpz_b_neurons_E) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_ex']['z']
                    axons_conn = syn_elms['Axon_ex']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

            se_fn_p_lpz_E = (
                "05-se-p_lpz_E-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_p_lpz_E, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.p_lpz_neurons_E) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_ex']['z']
                    axons_conn = syn_elms['Axon_ex']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

            # inhibitory neurons
            se_fn_lpz_c_I = (
                "05-se-lpz_c_I-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_lpz_c_I, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.lpz_c_neurons_I) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_in']['z']
                    axons_conn = syn_elms['Axon_in']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

            se_fn_lpz_b_I = (
                "05-se-lpz_b_I-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_lpz_b_I, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.lpz_b_neurons_I) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_in']['z']
                    axons_conn = syn_elms['Axon_in']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

            se_fn_p_lpz_I = (
                "05-se-p_lpz_I-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_p_lpz_I, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.p_lpz_neurons_I) if
                           stat['local']]
                for info in allinfo:
                    syn_elms = info[1]
                    axons = syn_elms['Axon_in']['z']
                    axons_conn = syn_elms['Axon_in']['z_connected']
                    dendrites_ex = syn_elms['Den_ex']['z']
                    dendrites_ex_conn = syn_elms['Den_ex']['z_connected']
                    dendrites_in = syn_elms['Den_in']['z']
                    dendrites_in_conn = syn_elms['Den_in']['z_connected']

                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        info[0],
                        axons, axons_conn,
                        dendrites_ex, dendrites_ex_conn,
                        dendrites_in, dendrites_in_conn),
                        file=f)

    def __dump_synaptic_weights(self):
        """Dump synaptic weights per rank.

        In a parallel simulation GetConnections only returns connections
        with *targets* on the MPI process executing the function.

        http://www.nest-simulator.org/cc/GetConnections/
        """
        current_sim_time = (str(nest.GetKernelStatus()['time']))

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsI)
        weightsIE = nest.GetStatus(conns, "weight")
        print("{}\t{}".format(
            current_sim_time,
            str(weightsIE).strip('[]').strip('()').replace(', ', '\t')),
            file=self.weights_fh_IE)
        if len(weightsIE) > self.num_synapses_IE:
            self.num_synapses_IE = len(weightsIE)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsI)
        weightsII = nest.GetStatus(conns, "weight")
        print("{}\t{}".format(
            current_sim_time,
            str(weightsII).strip('[]').strip('()').replace(', ', '\t')),
            file=self.weights_fh_II)
        if len(weightsII) > self.num_synapses_II:
            self.num_synapses_II = len(weightsII)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsE)
        weightsEI = nest.GetStatus(conns, "weight")
        print("{}\t{}".format(
            current_sim_time,
            str(weightsEI).strip('[]').strip('()').replace(', ', '\t')),
            file=self.weights_fh_EI)
        if len(weightsEI) > self.num_synapses_EI:
            self.num_synapses_EI = len(weightsEI)

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsE)
        weightsEE = nest.GetStatus(conns, "weight")
        print("{}\t{}".format(
            current_sim_time,
            str(weightsEE).strip('[]').strip('()').replace(', ', '\t')),
            file=self.weights_fh_EE)
        if len(weightsEE) > self.num_synapses_EE:
            self.num_synapses_EE = len(weightsEE)

    def dump_data(self):
        """Master datadump function."""
        logging.debug("Rank {}: Printing data to files".format(self.rank))
        self.__dump_synaptic_weights()
        self.__dump_ca_concentration()
        self.__dump_synaptic_elements_per_neurons()

    def close_files(self):
        """Close all files when the simulation is finished."""
        logging.debug("Rank {}: Closing open files".format(self.rank))
        # -1 is a unique index that marks this line. Since the other indexes
        # represent time, they cannot be negative
        print("-1\t{}".format(self.num_synapses_EE),
              file=self.weights_fh_EE)
        self.weights_fh_EE.close()

        print("-1\t{}".format(self.num_synapses_EI),
              file=self.weights_fh_EI)
        self.weights_fh_EI.close()
        print("-1\t{}".format(self.num_synapses_II),
              file=self.weights_fh_II)
        self.weights_fh_II.close()

        print("-1\t{}".format(self.num_synapses_IE),
              file=self.weights_fh_IE)
        self.weights_fh_IE.close()

        if self.is_str_p_enabled:
            if self.rank == 0:
                self.syn_new_fh_lpz_c_E.close()
                self.syn_new_fh_lpz_b_E.close()
                self.syn_new_fh_p_lpz_E.close()
                self.syn_new_fh_o_E.close()
                self.syn_del_fh_lpz_c_E.close()
                self.syn_del_fh_lpz_b_E.close()
                self.syn_del_fh_p_lpz_E.close()
                self.syn_del_fh_o_E.close()
                self.syn_new_fh_lpz_c_I.close()
                self.syn_new_fh_lpz_b_I.close()
                self.syn_new_fh_p_lpz_I.close()
                self.syn_new_fh_o_I.close()
                self.syn_del_fh_lpz_c_I.close()
                self.syn_del_fh_lpz_b_I.close()
                self.syn_del_fh_p_lpz_I.close()
                self.syn_del_fh_o_I.close()

    def enable_rewiring(self):
        """Enable the rewiring."""
        if not self.is_str_p_enabled:
            logging.critical("Structural plasticity isnt enabled!")
            logging.critical("Doing nothing")
            return 0

        self.is_rewiring_enabled = True
        if self.syn_del_strategy not in [
                "random", "distance", "weight"]:
            logging.critical(
                "INVALID SYNAPSE DELETION STRATEGY: {}".format(
                    self.syn_del_strategy))
            logging.critical("EXITING SIMULATION.")
            sys.exit(-1)
        if self.syn_form_strategy not in ["random", "distance"]:
            logging.critical(
                "INVALID SYNAPSE FORMATION STRATEGY: {}".format(
                    self.syn_form_strategy))
            logging.critical("EXITING SIMULATION.")
            sys.exit(-1)

        logging.info("Rank {}: REWIRING ENABLED".format(self.rank))

    def disable_rewiring(self):
        """Disable the rewiring."""
        if not self.is_str_p_enabled:
            logging.critical("Structural plasticity isnt enabled!")
            logging.critical("Doing nothing")
            return 0

        self.is_rewiring_enabled = False
        logging.info("Rank {}: REWIRING DISABLED".format(self.rank))

    def set_lpz_percent(self, lpz_percent):
        """Set up the network for deaff."""
        self.lpz_percent = lpz_percent
        logging.info("LPZ percent set to {}".format(self.lpz_percent))

    def __set_str_p_hom_point(self, ca_e, ca_i):
        """Set the new gaussian parameters for MSP."""
        self.eps_e = numpy.mean(ca_e)
        self.eps_i = numpy.mean(ca_i)
        # 0.4/0.7
        self.eta_ax_e = self.eps_e * 0.56
        self.eta_ax_i = self.eps_i * 0.56
        # 0.1/0.7
        self.eta_den_e = self.eps_e * 0.14
        self.eta_den_i = self.eps_i * 0.14

        new_growth_curve_axonal_E = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,  # max dz/dt (elements/ms)
            'continuous': False,
            'eta': self.eta_ax_e,
            'eps': self.eps_e
        }
        new_growth_curve_axonal_I = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,  # max dz/dt (elements/ms)
            'continuous': False,
            'eta': self.eta_ax_i,
            'eps': self.eps_i
        }
        new_growth_curve_dendritic_E = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,  # max dz/dt (elements/ms)
            'continuous': False,
            'eta': self.eta_den_e,
            'eps': self.eps_e
        }
        new_growth_curve_dendritic_I = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,  # max dz/dt (elements/ms)
            'continuous': False,
            'eta': self.eta_den_i,
            'eps': self.eps_i
        }

        new_structural_p_elements_E = {
            'Den_ex': new_growth_curve_dendritic_E,
            'Den_in': new_growth_curve_dendritic_I,
            'Axon_ex': new_growth_curve_axonal_E
        }

        new_structural_p_elements_I = {
            'Den_ex': new_growth_curve_dendritic_I,
            'Den_in': new_growth_curve_dendritic_I,
            'Axon_in': new_growth_curve_axonal_I
        }

        loc_e = [stat['global_id'] for stat in nest.GetStatus(self.neuronsE)
                 if stat['local']]
        loc_i = [stat['global_id'] for stat in nest.GetStatus(self.neuronsI)
                 if stat['local']]
        nest.SetStatus(loc_e, 'synaptic_elements_param',
                       new_structural_p_elements_E)
        nest.SetStatus(loc_i, 'synaptic_elements_param',
                       new_structural_p_elements_I)


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        format='%(funcName)s: %(lineno)d: %(levelname)s: %(message)s',
        level=logging.ERROR)

    step = False
    store_patterns = False
    deafferentate_network = True
    simulation = Sinha2016()

    # simulation setup
    # Setup network to handle plasticities
    # update of the network
    simulation.setup_plasticity(True, True)
    # set up deaff extent, and neuron sets
    simulation.set_lpz_percent(0.3)
    # set up neurons, connections, spike detectors, files
    simulation.prerun_setup(
        stabilisation_time=1500.,
        sp_update_interval=1500.,
        recording_interval=250.)
    # print em up
    simulation.print_simulation_parameters()
    logging.info("Rank {}: SIMULATION SETUP".format(simulation.rank))

    # initial setup
    logging.info("Rank {}: SIMULATION STARTED".format(simulation.rank))
    simulation.stabilise(label="Initial stabilisation")

    # Pattern related simulation
    if store_patterns:
        # control LPZs
        # pattern at centre of LPZ
        simulation.store_pattern_off_centre([0., 0.], True)
        # outside the LPZ
        simulation.store_pattern_with_centre([10000, 2000], 600, True)
        # store other pattern
        simulation.store_pattern_off_centre([0., 2000.0], True)

        # stabilise network after storing patterns
        simulation.stabilise(label="Pattern stabilisation")
    # Set homoeostatic structural plasticity parameters to whatever the network
    # has achieved now
    simulation.invoke_metaplasticity()
    # Enable structural plasticity for repair #
    simulation.enable_rewiring()

    #  Stabilise with both plasticities active
    #  update time windows
    """
    simulation.update_time_windows(stabilisation_time=1000.,
                                   sp_update_interval=0.1,
                                   recording_interval=100.)
    simulation.stabilise()
    """

    if deafferentate_network:
        # Deaff network
        simulation.deaff_network()
        # zoom in on post deaff phase
        simulation.update_time_windows(stabilisation_time=50.,
                                       sp_update_interval=0.1,
                                       recording_interval=1.)
        simulation.stabilise()

        simulation.update_time_windows(stabilisation_time=2000.,
                                       sp_update_interval=0.1,
                                       recording_interval=250.)
        simulation.stabilise()

    if store_patterns:
        # recall stored and tracked pattern
        simulation.recall_pattern(50, 1)
        simulation.recall_pattern(50, 2)
        simulation.recall_pattern(50, 3)

    nest.Cleanup()
    simulation.close_files()
    logging.info("Rank {}: SIMULATION FINISHED SUCCESSFULLY".format(
        simulation.rank))
