#!/usr/bin/env python3
"""
NEST simulation code for my PhD research.

File: Sinha2016.py

Copyright 2019 Ankur Sinha
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
import mpi4py
from mpi4py import MPI
import logging
import operator
import time


mpi4py.rc.errors = 'fatal'


class Sinha2016:

    """Simulations for my PhD 2016."""

    def __init__(self):
        """Initialise variables."""
        self.comm = MPI.COMM_WORLD
        logging.info("MPI Backend:\n{}".format(MPI.Get_library_version()))
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
        self.syn_del_strategy = "weight"
        # "random" or "distance"
        self.syn_form_strategy = "distance"

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'STIM': 1000, 'Poisson': 1}
        # pattern percent of E neurons
        self.pattern_percent = .1
        # recall percent of pattern
        self.recall_percent = .25

        self.populations['P'] = int(self.pattern_percent *
                                    self.populations['E'])
        self.populations['R'] = int(self.recall_percent *
                                    self.populations['P'])

        # location bits
        self.colsE = 80
        self.colsI = 40
        self.neuronal_distE = 150  # micro metres
        self.neuronal_distI = 2 * self.neuronal_distE  # micro metres
        self.location_sd = 15  # micro metres
        # SD = w_mul * neuron_distE
        self.w_mul_E = 8.
        self.w_mul_I = 24.
        self.max_p_E = 0.8
        self.max_p_I = 0.3
        self.location_tree = None
        self.lpz_percent = 0.5
        # to calculate distances as if we're using a toroid
        # to not run into edge effects
        self.network_width = 0.
        self.network_height = 0.

        # structural plasticity bits
        # not steps since we're not using it in NEST. This is for our manual
        # updates
        self.sp_update_interval = 1000.  # seconds
        # time recall stimulus is enabled for
        self.recall_duration = 1000.  # ms
        # homoeostatic stable points for E and I neurons
        # before structural plasticity is enabled, these will be updated
        self.eps_den_e_e = 0.7
        self.eps_den_i_e = 0.7
        self.eps_den_e_i = 0.7
        self.eps_den_i_i = 0.7
        self.eps_ax_e = 0.7
        self.eps_ax_i = 0.7

        self.eta_ax_e = 0.3
        self.eta_ax_i = 0.3
        self.eta_den_e_e = 0.1
        self.eta_den_e_i = 0.1
        self.eta_den_i_e = 0.1
        self.eta_den_i_i = 0.1

        # maximum value of dz/dt per time step
        # z += dz
        # base value
        self.nu = 3e-5
        # value for excitatory neurons
        self.nu_e = self.nu
        self.nu_den_e_e = 1 * self.nu_e
        self.nu_den_e_i = 10 * self.nu_e
        # 8000 excitatory neurons provide input to 10000 neurons of the
        # network, so they must be slightly faster than the post-synaptic
        # elements
        self.nu_ax_e = 50 * self.nu

        # post-synaptic elements of inhibitory neurons react slower to activity
        # changes than excitatory neurons so that the inhibitory network
        # reacts to the dynamics of the excitatory network.
        self.nu_i = self.nu
        self.nu_den_i_e = 1 * self.nu_i
        self.nu_den_i_i = 1 * self.nu_i
        # 2000 inhibitory neurons inhibit 10000 neurons of the network, so the
        # inhibitory axon must act at a faster rate
        self.nu_ax_i = 1000 * self.nu

        # z_vacant is multiplied by this tau at each time step to decay it.
        # z -= z_vacant * tau
        # 1% of free synaptic elements are lost at each time step
        self.tau_ax_e = 0.01
        self.tau_den_e_e = 0.01
        self.tau_den_e_i = 0.01

        self.tau_ax_i = 0.01
        self.tau_den_i_e = 0.01
        self.tau_den_i_i = 0.01

        self.omega_ax_e = 0.01
        self.omega_den_e_e = 0.4
        self.omega_den_e_i = 0.04

        self.omega_ax_i = 0.0004
        self.omega_den_i_e = 0.4
        self.omega_den_i_i = 0.4

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
        self.weightIE = -0.000000000000000001  # initial weight for plastic IE
        self.weightIEmax = -20.  # simulations without limits suggest this
        self.weightSD_IEmax = self.weightIEmax/5.
        self.weightSD_EE = self.weightEE/5
        self.weightSD_EI = self.weightEI/5
        self.weightSD_II = self.weightII/5
        self.weightSD_IE = self.weightIE/5
        self.weightPatternEE = self.wbar * 5.
        self.weightExtE = 8.
        self.weightExtI = 12.
        self.stability_threshold_I = 100000.
        self.stability_threshold_E = 100000.

        self.etaIE = 0.05
        self.alphaIE = .12
        self.tauIE = 20.

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

        # a shuffled list of neurons for connection updates so that we don't
        # always iterate in ascending order of GID---if we did, neurons with
        # lower GIDs would have a chance to form connections first.
        self.shuffled_neurons = []

        # record when network was deaffed
        self.deaffed_at = 0.

        random.seed(42)
        numpy.random.seed(42)

    def __get_distance_toroid(self, source, destination):
        """Get distance between a pair of neurons on our toroid

        :source: source neuron
        :destination: destination neuron
        :returns: distance between neurons if they were on a toroid

        """
        source_loc = numpy.array(
            self.location_tree.data[source - self.neuronsE[0]])
        dest_loc = numpy.array(
            self.location_tree.data[destination - self.neuronsE[0]])

        delta_x = abs(source_loc[0] - dest_loc[0])
        if delta_x > self.network_width/2:
            delta_x = self.network_width - delta_x

        delta_y = abs(source_loc[1] - dest_loc[1])
        if delta_y > self.network_height/2:
            delta_y = self.network_height - delta_y

        distance = math.hypot(delta_x, delta_y)
        return distance

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
            new_growth_curve_axonal_E = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_ax_e,
                'continuous': False,
                'eta': self.eta_ax_e,
                'eps': self.eps_ax_e,
                'omega': self.omega_ax_e
            }
            new_growth_curve_axonal_I = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_ax_i,
                'continuous': False,
                'eta': self.eta_ax_i,
                'eps': self.eps_ax_i,
                'omega': self.omega_ax_i
            }
            new_growth_curve_dendritic_E_e = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_den_e_e,
                'continuous': False,
                'eta': self.eta_den_e_e,
                'eps': self.eps_den_e_e,
                'omega': self.omega_den_e_e
            }
            new_growth_curve_dendritic_E_i = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_den_e_i,
                'continuous': False,
                'eta': self.eta_den_e_i,
                'eps': self.eps_den_e_i,
                'omega': self.omega_den_e_i
            }
            new_growth_curve_dendritic_I_e = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_den_i_e,
                'continuous': False,
                'eta': self.eta_den_i_e,
                'eps': self.eps_den_i_e,
                'omega': self.omega_den_i_e
            }
            new_growth_curve_dendritic_I_i = {
                'growth_curve': "gaussian",
                'growth_rate': 0.,
                'tau_vacant': self.tau_den_i_i,
                'continuous': False,
                'eta': self.eta_den_i_i,
                'eps': self.eps_den_i_i,
                'omega': self.omega_den_i_i
            }

            self.structural_p_elements_E = {
                'Den_ex': new_growth_curve_dendritic_E_e,
                'Den_in': new_growth_curve_dendritic_E_i,
                'Axon_ex': new_growth_curve_axonal_E
            }

            self.structural_p_elements_I = {
                'Den_ex': new_growth_curve_dendritic_I_e,
                'Den_in': new_growth_curve_dendritic_I_i,
                'Axon_in': new_growth_curve_axonal_I
            }

        # see the aif source for symbol definitions
        self.neuronDict = {'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.,
                           'beta_Ca': 0.10, 'tau_Ca': 50000.
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

        self.shuffled_neurons = list(self.neuronsE + self.neuronsI)
        # when shuffling, set a constant seed for the Random generator so that
        # all process get identical results. This is necessary when making
        # connection updates.
        random.Random(23).shuffle(self.shuffled_neurons)

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

        self.network_width = (locations[-1][0] - locations[0][0])
        self.network_height = (locations[-1][1] - locations[0][1])

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
        self.o_neurons_E = list(set(self.neuronsE) - set(with_p_lpz_neurons_E))

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
        self.o_neurons_I = list(set(self.neuronsI) - set(with_p_lpz_neurons_I))

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

    def __get_synapses_to_form(self, sources, destinations, sparsity):
        """
        Find prospective synaptic connections between sets of neurons.

        Since structural plasticity does not permit sparse connections, I'm
        going to try to manually find the right number of syapses and connect
        neurons to get a certain sparsity.

        These connections are completely random. They do not depend on
        distance or any other such parameter.

        :sources: set of source neurons
        :destinations: set of destination neurons
        :sparsity: probability of forming a connection between a pair

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
            logging.critical(
                "Rank {}: Neither plasticity is enabled.  Exiting.".format(
                    self.rank)
            )
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
                                  'weight': -1.,
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
                                  'weight': self.weightIE,
                                  'Wmax': self.weightIEmax,
                                  'alpha': self.alphaIE, 'eta': self.etaIE,
                                  'tau': self.tauIE,
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
                              'weight': self.weightIE,
                              'Wmax': self.weightIEmax,
                              'alpha': self.alphaIE, 'eta': self.etaIE,
                              'tau': self.tauIE}

    def __create_initial_connections(self):
        """Initially connect various neuron sets."""
        nest.Connect(self.poissonExt, self.neuronsE,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtE})
        nest.Connect(self.poissonExt, self.neuronsI,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExtI})

        # if synaptic plasticity is enabled
        # setup connections using Nest primitives
        # we cannot use pairwise_bernoulli and so on because we want our
        # connections to be distance dependent. I.e., neurons closer to each
        # other are more likely to connect.
        if self.is_syn_p_enabled:
            conndict = {'rule': 'all_to_all'}
            logging.info("Rank {}: Setting up EE connections.".format(
                self.rank
            ))
            start_time = time.clock()
            max_num = (len(self.neuronsE) * len(self.neuronsE) * self.sparsity)
            indegreeEE = int(len(self.neuronsE)*self.sparsity)
            # shuffle the list each time so that the order in which prospective
            # targets are presented to the method is changed. Otherwise, if the
            # same set is passed to the method each time, even the shuffle call
            # in the method will return the same sequence, since it is run with
            # the same seed (fixed seed so that all ranks return the same
            # sequence).
            for nrn in self.neuronsE:
                sources = self.__get_nearest_ps_gaussian(
                    nrn, self.neuronsE,
                    int(random.gauss(indegreeEE, 0.1*indegreeEE)),
                    w_mul=self.w_mul_E, max_p=self.max_p_E)
                nest.Connect(sources, [nrn],
                             syn_spec=self.synDictEE,
                             conn_spec=conndict)
            conns = nest.GetConnections(source=self.neuronsE,
                                        target=self.neuronsE)
            for acon in conns:
                nest.SetStatus(
                    [acon], {
                        'weight': random.gauss(
                            self.weightEE, self.weightSD_EE
                        )
                    }
                )
            end_time = time.clock()
            logging.info(
                "Rank {}: {}/{} EE connections set up in {}s.".format(
                    self.rank, len(conns), int(max_num/self.comm.Get_size()),
                    (end_time - start_time)/60.))

            logging.info("Rank {}: Setting up EI connections.".format(
                self.rank
            ))
            start_time = end_time
            max_num = (len(self.neuronsE) * len(self.neuronsI) * self.sparsity)
            indegreeEI = int(len(self.neuronsE)*self.sparsity)
            for nrn in self.neuronsI:
                sources = self.__get_nearest_ps_gaussian(
                    nrn, self.neuronsE,
                    int(random.gauss(indegreeEI, 0.1*indegreeEI)),
                    w_mul=self.w_mul_E,
                    max_p=self.max_p_E)
                nest.Connect(sources, [nrn],
                             syn_spec=self.synDictEI,
                             conn_spec=conndict)
            conns = nest.GetConnections(source=self.neuronsE,
                                        target=self.neuronsI)
            for acon in conns:
                nest.SetStatus(
                    [acon], {
                        'weight': random.gauss(
                            self.weightEI, self.weightSD_EI
                        )
                    }
                )
            end_time = time.clock()
            logging.info(
                "Rank {}: {}/{} EI connections set up in {}s.".format(
                    self.rank, len(conns), int(max_num/self.comm.Get_size()),
                    (end_time - start_time)/60.))

            logging.info("Rank {}: Setting up II connections.".format(
                self.rank
            ))
            start_time = end_time
            max_num = (len(self.neuronsI) * len(self.neuronsI) * self.sparsity)
            indegreeII = int(len(self.neuronsI)*self.sparsity)
            for nrn in self.neuronsI:
                sources = self.__get_nearest_ps_gaussian(
                    nrn, self.neuronsI,
                    int(random.gauss(indegreeII, 0.1*indegreeII)),
                    w_mul=self.w_mul_I,
                    max_p=self.max_p_I)
                nest.Connect(sources, [nrn],
                             syn_spec=self.synDictII,
                             conn_spec=conndict)
            conns = nest.GetConnections(source=self.neuronsI,
                                        target=self.neuronsI)
            for acon in conns:
                nest.SetStatus(
                    [acon], {
                        'weight': random.gauss(
                            self.weightII,
                            self.weightSD_II
                        )
                    }
                )
            end_time = time.clock()
            logging.info(
                "Rank{}: {}/{} II connections set up in {}s.".format(
                    self.rank, len(conns), int(max_num/self.comm.Get_size()),
                    (end_time - start_time)/60.))

            logging.info("Rank {}: Setting up IE connections.".format(
                self.rank
            ))
            start_time = end_time
            max_num = (len(self.neuronsI) * len(self.neuronsE) * self.sparsity)
            indegreeIE = int(len(self.neuronsI)*self.sparsity)
            for nrn in self.neuronsE:
                sources = self.__get_nearest_ps_gaussian(
                    nrn, self.neuronsI,
                    int(random.gauss(indegreeIE, 0.1*indegreeIE)),
                    w_mul=self.w_mul_I,
                    max_p=self.max_p_I)
                nest.Connect(sources, [nrn],
                             syn_spec=self.synDictIE,
                             conn_spec=conndict)
            conns = nest.GetConnections(source=self.neuronsI,
                                        target=self.neuronsE)
            for acon in conns:
                wmax = random.gauss(self.weightIEmax,
                                    self.weightSD_IEmax)
                while wmax >= 0.:
                    wmax = random.gauss(self.weightIEmax,
                                        self.weightSD_IEmax)
                nest.SetStatus(
                    [acon], {
                        'weight': self.weightIE,
                        'Wmax': wmax
                    }
                )
            end_time = time.clock()
            logging.info(
                "Rank{}: {}/{} IE connections set up in {}s.".format(
                    self.rank, len(conns), int(max_num/self.comm.Get_size()),
                    (end_time - start_time)/60.))
        else:
            logging.info("Synaptic plasticity not enabled." +
                         "Not setting up any synapses.")

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

    def __set_str_p_params(self):
        """Set the new gaussian parameters for MSP."""

        # multipliers
        eps_ax_e_mul = 1.75
        eps_den_e_e_mul = 1.0
        eps_den_e_i_mul = 3.50
        eta_ax_e_mul = 1.0
        eta_den_e_e_mul = 0.25
        eta_den_e_i_mul = 1.0

        list_e = numpy.array(
            [[stat['global_id'], stat['Ca']] for stat in
             nest.GetStatus(self.neuronsE) if stat['local']])
        list_i = numpy.array(
            [[stat['global_id'], stat['Ca']] for stat in
             nest.GetStatus(self.neuronsI) if stat['local']])

        new_structural_p_elements_E = []
        for [gid, ca] in list_e:
            eps_ax_e = ca * eps_ax_e_mul
            eps_den_e_e = ca * eps_den_e_e_mul
            eps_den_e_i = ca * eps_den_e_i_mul
            eta_ax_e = ca * eta_ax_e_mul
            eta_den_e_e = ca * eta_den_e_e_mul
            eta_den_e_i = ca * eta_den_e_i_mul

            new_growth_curve_axonal_E = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_ax_e,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_ax_e,
                'continuous': False,
                'eta': eta_ax_e,
                'eps': eps_ax_e,
                'omega': self.omega_ax_e
            }

            new_growth_curve_dendritic_E_e = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_den_e_e,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_den_e_e,
                'continuous': False,
                'eta': eta_den_e_e,
                'eps': eps_den_e_e,
                'omega': self.omega_den_e_e
            }
            new_growth_curve_dendritic_E_i = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_den_e_i,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_den_e_i,
                'continuous': False,
                'eta': eta_den_e_i,
                'eps': eps_den_e_i,
                'omega': self.omega_den_e_i
            }

            new_structural_p_elements_E.append({
                'Den_ex': new_growth_curve_dendritic_E_e,
                'Den_in': new_growth_curve_dendritic_E_i,
                'Axon_ex': new_growth_curve_axonal_E
            })
        gids_e = tuple(list_e[:, 0].astype(int))
        nest.SetStatus(gids_e, 'synaptic_elements_param',
                       new_structural_p_elements_E)

        # For I
        eps_ax_i_mul = 1.0
        eps_den_i_e_mul = 1.0
        eps_den_i_i_mul = 3.50
        eta_ax_i_mul = 0.25
        eta_den_i_e_mul = 0.25
        eta_den_i_i_mul = 1.0

        new_structural_p_elements_I = []
        for [gid, ca] in list_i:
            eps_ax_i = ca * eps_ax_i_mul
            eps_den_i_e = ca * eps_den_i_e_mul
            eps_den_i_i = ca * eps_den_i_i_mul
            eta_ax_i = ca * eta_ax_i_mul
            eta_den_i_e = ca * eta_den_i_e_mul
            eta_den_i_i = ca * eta_den_i_i_mul

            new_growth_curve_axonal_I = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_ax_i,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_ax_i,
                'continuous': False,
                'eta': eta_ax_i,
                'eps': eps_ax_i,
                'omega': self.omega_ax_i
            }
            new_growth_curve_dendritic_I_e = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_den_i_e,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_den_i_e,
                'continuous': False,
                'eta': eta_den_i_e,
                'eps': eps_den_i_e,
                'omega': self.omega_den_i_e
            }
            new_growth_curve_dendritic_I_i = {
                'growth_curve': "gaussian",
                'growth_rate': self.nu_den_i_i,  # max dz/dt (elements/ms)
                'tau_vacant': self.tau_den_i_i,
                'continuous': False,
                'eta': eta_den_i_i,
                'eps': eps_den_i_i,
                'omega': self.omega_den_i_i
            }
            new_structural_p_elements_I.append({
                'Den_ex': new_growth_curve_dendritic_I_e,
                'Den_in': new_growth_curve_dendritic_I_i,
                'Axon_in': new_growth_curve_axonal_I
            })

        gids_i = tuple(list_i[:, 0].astype(int))
        nest.SetStatus(gids_i, 'synaptic_elements_param',
                       new_structural_p_elements_I)

        # Network means
        # Purely for printing and graphing only
        # For E
        mean_ca_e = numpy.mean(list_e[:, 1])
        mean_ca_i = numpy.mean(list_i[:, 1])
        self.eps_ax_e = mean_ca_e * eps_ax_e_mul
        self.eps_den_e_e = mean_ca_e * eps_den_e_e_mul
        self.eps_den_e_i = mean_ca_e * eps_den_e_i_mul
        self.eta_ax_e = mean_ca_e * eta_ax_e_mul
        self.eta_den_e_e = mean_ca_e * eta_den_e_e_mul
        self.eta_den_e_i = mean_ca_e * eta_den_e_i_mul

        # For I
        self.eps_ax_i = mean_ca_i * eps_ax_i_mul
        self.eps_den_i_e = mean_ca_i * eps_den_i_e_mul
        self.eps_den_i_i = mean_ca_i * eps_den_i_i_mul
        self.eta_ax_i = mean_ca_i * eta_ax_i_mul
        self.eta_den_i_e = mean_ca_i * eta_den_i_e_mul
        self.eta_den_i_i = mean_ca_i * eta_den_i_i_mul

        logging.debug("Rank {}: Updated growth curves.".format(self.rank))

    def setup_simulation(self):
        """Setup the common simulation things."""
        # Nest stuff
        start_time = time.clock()
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
        end_time = time.clock()
        time_taken = end_time - start_time
        logging.info("Rank {}: SIMULATION SETUP took {}s".format(
            simulation.rank, time_taken/60))

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
        logging.debug("Rank {}: Got {} local neurons".format(
             self.rank, len(local_neurons)))

        lneurons = ()
        lneurons = nest.GetStatus(local_neurons, ['global_id',
                                                  'synaptic_elements'])
        self.comm.barrier()
        # returns a list of sets - one set from each rank
        ranksets = self.comm.allgather(lneurons)
        logging.debug("Rank {}: Got {} ranksets".format(
            self.rank, len(ranksets)))

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

                if math.isnan(source_elms_con):
                    source_elms_con = 0
                if math.isnan(source_elms_total):
                    source_elms_total = 0

                target_elms_con_ex = synelms['Den_ex']['z_connected']
                target_elms_con_in = synelms['Den_in']['z_connected']
                target_elms_total_ex = synelms['Den_ex']['z']
                target_elms_total_in = synelms['Den_in']['z']

                if math.isnan(target_elms_con_ex):
                    target_elms_con_ex = 0
                if math.isnan(target_elms_con_in):
                    target_elms_con_in = 0
                if math.isnan(target_elms_total_ex):
                    target_elms_total_ex = 0
                if math.isnan(target_elms_total_in):
                    target_elms_total_in = 0

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
            "Rank {}: Got synaptic elements for {} neurons.".format(
                self.rank, len(synaptic_elms)))
        return synaptic_elms

    def __get_random_ps_to_delete(self, options, num_required):
        """Get random partners from options.

        This is to be used when all the targets are connected to a neuron with
        the same weight, so one cannot differentiate between them based on
        weights. They have the same likelihood of deletion, so we randomly pick
        partners to remove.

        So, the weight is disregarded in this function.

        :options: targets that can be lost
        :num_required: number of targets to lose
        :returns: chosen_targets to remove

        """
        logging.debug("Rank {}: Returning weakest links randomly".format(
            self.rank
        ))
        targets = [nid for nid, weight in options]

        # if there aren't enough candidates, return them all
        if len(targets) < num_required:
            return targets

        chosen_options = numpy.random.choice(targets, num_required,
                                             replace=False)
        return list(chosen_options)

    def __get_weakest_ps_gaussian(self, options, num_required,
                                  threshold=10000.):
        """Choose partners to delete based on weight of connections.

        However, instead of simply picking the weakest ones, I first calculate
        their probability of deletion and pick ones to delete stochastically.

        Note that the signs of w and threshold need to have the right signs,
        this method does not check that bit.

        :options: options to pick from as [[nid, weight]]
        :num_required: number of options required
        :threshold: only consider synapses weaker than this conductance
        :returns: chosen options

        """
        logging.debug("Rank {}:Returning weakest links stochastically".format(
            self.rank
        ))
        candidates = [[n, w] for n, w in options if w < threshold]
        sorted_options = sorted(candidates, key=operator.itemgetter(1))

        targets = []
        for nid, w in sorted_options:
            probability = (math.exp(-1*((w**2)/(threshold*2)**2)))
            # weaker synapses have higher probability of removal
            if random.random() <= probability:
                targets.append(nid)
            if len(targets) >= num_required:
                return targets

        return targets

    def __get_weakest_ps(self, options, num_required, threshold=10000.):
        """Choose partners to delete based on weight of connections.

        Note that the signs of w and threshold need to have the right signs,
        this method does not check that bit.

        :options: options to pick from as [[nid, weight]]
        :num_required: number of options required
        :threshold: only consider synapses weaker than this conductance
        :returns: chosen options

        """
        logging.debug("Rank {}: Returning weakest links".format(self.rank))
        candidates = [[n, w] for n, w in options if w < threshold]
        if len(candidates) < num_required:
            weakest_options = [nid for nid, weight in candidates]
            return weakest_options

        sorted_options = sorted(candidates, key=operator.itemgetter(1))
        weakest_options = [nid for nid, weight in
                           sorted_options[0:num_required]]

        return weakest_options

    def __get_farthest_ps(self, anchor, options, num_required):
        """Choose farthest partners.

        :anchor: source neuron
        :options: options to choose from
        :num_required: number of options needed
        :returns: a list of chosen options

        """
        # if we don't have enough options to choose, simply return whatever is
        # available
        logging.debug("Rank {}: Returning farthest partners".format(self.rank))
        if len(options) < num_required:
            return options

        options_with_distances = []
        for opt in options:
            distance = self.__get_distance_toroid(anchor, opt)
            options_with_distances.append([opt, distance])

        sorted_options = sorted(options_with_distances,
                                key=operator.itemgetter(1), reverse=True)
        farthest_options = [nid for nid, distance in
                            sorted_options[0:num_required]]

        return farthest_options

    def __get_nearest_ps_prob(self, source, options, probability):
        """Choose nearest partners but pick them with a probability.
        The probability does not depend on the distance between neurons here.

        Do not permit autapses.

        :source: source neuron
        :options: options to choose from
        :probability: probability of picking a partner
        :returns: list of chosen nearest partners

        """
        # inefficient, but works.
        max_num_required = int(len(options) * probability)
        options_with_distances = []
        for opt in options:
            if opt == source:
                continue
            distance = self.__get_distance_toroid(source, opt)
            options_with_distances.append([opt, distance])

        sorted_options = sorted(options_with_distances,
                                key=operator.itemgetter(1))
        chosen_options = []
        for nrn, distance in sorted_options:
            if random.random() <= probability:
                chosen_options.append(nrn)
                max_num_required -= 1
            # when max_num_required is zero, return
            if not max_num_required:
                return chosen_options

        # otherwise just return how many we got after traversing the whole
        # option list
        return chosen_options

    def __get_nearest_ps_gaussian(self, source, options, num_required,
                                  w_mul=10., max_p=1.):
        """
        Choose nearest partners but with a gaussian kernel, within a bounded
        distance.

        :source: source neuron
        :w_mul: width of Gaussian = w_mul x neuronal_distE
        :max_p: maximum probability
        :options: options to choose from
        :num_required: number of partners needed
        :returns: list of chosen nearest partners

        """
        logging.debug(
            "Rank {}: Returning partners (gaussian probability)".format(
                self.rank))
        options_with_distances = []
        for opt in options:
            if opt == source:
                continue
            distance = self.__get_distance_toroid(source, opt)
            options_with_distances.append([opt, distance])

        sorted_options = sorted(options_with_distances,
                                key=operator.itemgetter(1))
        chosen_options = []
        for opt, distance in sorted_options:
            probability = (
                max_p *
                math.exp(-1.*((distance**2)/((w_mul*self.neuronal_distE)**2)))
            )
            if random.random() <= probability:
                chosen_options.append(opt)
                num_required -= 1
            # when num_required hits 0, return
            if not num_required:
                return chosen_options

        return chosen_options

    def __get_nearest_ps(self, source, options, num_required):
        """Choose nearest partners.

        :source: source neuron
        :options: options to choose from
        :num_required: number of partners needed
        :returns: list of chosen nearest partners

        """
        logging.debug("Rank {}: Returning nearest partners".format(self.rank))
        if len(options) < num_required:
            return options

        # inefficient, but works.
        options_with_distances = []
        for opt in options:
            distance = self.__get_distance_toroid(source, opt)
            options_with_distances.append([opt, distance])

        sorted_options = sorted(options_with_distances,
                                key=operator.itemgetter(1))
        nearest_options = [nid for nid, distance in
                           sorted_options[0:num_required]]

        return nearest_options

    def __delete_connections_from_pre(self, synelms):
        """Delete connections when the neuron is a source."""
        logging.debug
        ("Rank {}: Deleting from pre using '{}' deletion strategy".format(
            self.rank, self.syn_del_strategy))
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
        for nrn in self.shuffled_neurons:
            gid = nrn
            elms = synelms[nrn]
            partner = 0  # for exception
            total_synapses_this_gid = 0
            syn_del_this_gid = 0
            try:
                # excitatory neurons as sources
                if 'Axon_ex' in elms and elms['Axon_ex'] < 0:
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
                                [conns[i][1], abs(weights[i])])
                    else:
                        for acon in conns:
                            localtargets.append(acon[1])

                    self.comm.barrier()
                    alltargets = self.comm.allgather(localtargets)
                    targets = [t for sublist in alltargets for t in sublist]
                    total_synapses_this_gid = len(targets)
                    if len(targets) > 0:
                        # this is where the selection logic is
                        if self.syn_del_strategy == "random":
                            # Doesn't merit a new method
                            if len(targets) > int(abs(elms['Axon_ex'])):
                                chosen_targets = random.sample(
                                    targets, int(abs(elms['Axon_ex'])))
                            else:
                                chosen_targets = targets
                        elif self.syn_del_strategy == "distance":
                            chosen_targets = self.__get_farthest_ps(
                                gid, targets, int(abs(elms['Axon_ex'])))
                        elif self.syn_del_strategy == "weight":
                            # need to fetch targets from [nid, w], so use a
                            # method for clarity
                            chosen_targets = self.__get_weakest_ps_gaussian(
                                targets, int(abs(elms['Axon_ex'])),
                                threshold=self.stability_threshold_E
                            )

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
                                [connsToI[i][1], abs(weightsToI[i])])
                        for i in range(0, len(connsToE)):
                            localtargetsE.append(
                                [connsToE[i][1], abs(weightsToE[i])])
                    else:
                        for acon in connsToI:
                            localtargetsI.append(acon[1])
                        for acon in connsToE:
                            localtargetsE.append(acon[1])

                    self.comm.barrier()
                    alltargetsI = self.comm.allgather(localtargetsI)
                    alltargetsE = self.comm.allgather(localtargetsE)

                    targetsI = [t for sublist in alltargetsI for t in sublist]
                    targetsE = [t for sublist in alltargetsE for t in sublist]

                    total_synapses_this_gid = (len(targetsE) + len(targetsI))

                    if (total_synapses_this_gid) > 0:
                        # this is where the selection logic is
                        if self.syn_del_strategy == "random":
                            if (total_synapses_this_gid) > \
                                    int(abs(elms['Axon_in'])):
                                # Doesn't merit a new method
                                chosen_targets = random.sample(
                                    (targetsE + targetsI),
                                    int(abs(elms['Axon_in'])))
                            else:
                                chosen_targets = (targetsE + targetsI)
                        elif self.syn_del_strategy == "distance":
                            chosen_targets = self.__get_farthest_ps(
                                gid, (targetsE + targetsI),
                                int(abs(elms['Axon_in'])))
                        elif self.syn_del_strategy == "weight":
                            # use threshold for I* synapses.
                            chosen_targets = self.__get_weakest_ps_gaussian(
                                (targetsE + targetsI),
                                int(abs(elms['Axon_in'])),
                                threshold=self.stability_threshold_I)
                            # strip the weights from the list now since we
                            # compare with these later
                            targetsE = [nid for nid, weight in targetsE]
                            targetsI = [nid for nid, weight in targetsI]

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
            except Exception as e:
                logging.critical("Some other exception: {}".format(str(e)))
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
            "Rank {}: Deleting from post using '{}' deletion strategy".format(
                self.rank, self.syn_del_strategy))
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

        for nrn in self.shuffled_neurons:
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
                                [conns[i][0], abs(weights[i])])
                    else:
                        for acon in conns:
                            localsources.append(acon[0])

                    self.comm.barrier()
                    allsources = self.comm.allgather(localsources)
                    sources = [s for sublist in allsources for s in sublist]
                    total_synapses_this_gid += len(sources)

                    if len(sources) > 0:
                        if self.syn_del_strategy == "random":
                            if len(sources) > int(abs(elms['Den_ex'])):
                                chosen_sources = random.sample(
                                    sources, int(abs(elms['Den_ex'])))
                            else:
                                chosen_sources = sources
                        elif self.syn_del_strategy == "distance":
                            chosen_sources = self.__get_farthest_ps(
                                gid, sources, int(abs(elms['Den_ex'])))
                        elif self.syn_del_strategy == "weight":
                            # need to strip [nid, w] to get targets, so using a
                            # different function for clarity
                            chosen_sources = self.__get_weakest_ps_gaussian(
                                sources, int(abs(elms['Den_ex'])),
                                threshold=self.stability_threshold_E
                            )

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
                        conns = nest.GetConnections(
                            target=[gid], synapse_model='static_synapse_in')
                        localsources = []
                        chosen_sources = []

                        if self.syn_del_strategy == "weight":
                            # also need to store weight
                            weights = nest.GetStatus(conns, "weight")
                            for i in range(0, len(conns)):
                                localsources.append(
                                    [conns[i][0], abs(weights[i])])
                        else:
                            for acon in conns:
                                localsources.append(acon[0])

                        self.comm.barrier()
                        allsources = self.comm.allgather(localsources)
                        sources = [s for sublist in allsources for s in
                                   sublist]
                        total_synapses_this_gid += len(sources)

                        if len(sources) > 0:
                            if self.syn_del_strategy == "random":
                                if len(sources) > int(abs(elms['Den_in'])):
                                    chosen_sources = random.sample(
                                        sources, int(abs(elms['Den_in'])))
                                else:
                                    chosen_sources = sources
                            elif self.syn_del_strategy == "distance":
                                chosen_sources = self.__get_farthest_ps(
                                    gid, sources, int(abs(elms['Den_in'])))
                            elif self.syn_del_strategy == "weight":
                                # II synapses, use threshold, even though they
                                # are all of the same weight, because we'll set
                                # the threshold to 0 to disable deletion of II
                                # synapses complete, or we'll set it to a
                                # really high value to disable thresholding.
                                chosen_sources = (
                                    self.__get_weakest_ps_gaussian(
                                        sources, int(abs(elms['Den_in'])),
                                        threshold=self.stability_threshold_I)
                                )

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
                                localsources.append([conns[i][0],
                                                     abs(weights[i])])
                        else:
                            # otherwise only sids
                            for acon in conns:
                                localsources.append(acon[0])
                        self.comm.barrier()
                        allsources = self.comm.allgather(localsources)
                        sources = [s for sublist in allsources for s in
                                   sublist]
                        total_synapses_this_gid += len(sources)

                        if len(sources) > 0:
                            if self.syn_del_strategy == "random":
                                if len(sources) > int(abs(elms['Den_in'])):
                                    chosen_sources = random.sample(
                                        sources, int(abs(elms['Den_in'])))
                                else:
                                    chosen_sources = sources
                            elif self.syn_del_strategy == "distance":
                                chosen_sources = self.__get_farthest_ps(
                                    gid, sources, int(abs(elms['Den_in'])))
                            elif self.syn_del_strategy == "weight":
                                # IE synapses, so use threshold
                                chosen_sources = (
                                    self.__get_weakest_ps_gaussian(
                                        sources, int(abs(elms['Den_in'])),
                                        threshold=self.stability_threshold_I)
                                )

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
            except Exception as e:
                logging.critical("Some other exception: {}".format(str(e)))
                raise

        logging.debug(
            "{} of {} connections deleted from post".format(
                (syn_del_lpz_c_E + syn_del_lpz_b_E +
                 syn_del_p_lpz_E + syn_del_lpz_c_I +
                 syn_del_lpz_b_I + syn_del_p_lpz_I +
                 syn_del_o_E + syn_del_o_I),
                total_synapses))

    def __create_new_connections_from_post(self, synelms):
        """Create new connections from the post-synaptic side."""
        logging.debug("Rank {}: Creating using the {} strategy".format(
            self.rank, self.syn_form_strategy))
        syn_new_lpz_c_E = 0
        syn_new_lpz_b_E = 0
        syn_new_p_lpz_E = 0
        syn_new_o_E = 0
        syn_new_lpz_c_I = 0
        syn_new_lpz_b_I = 0
        syn_new_p_lpz_I = 0
        syn_new_o_I = 0
        current_sim_time = (str(nest.GetKernelStatus()['time']))

        for nrn in self.shuffled_neurons:
            syn_new_this_gid = 0
            total_options_this_gid = 0
            gid = nrn
            elms = synelms[nrn]
            chosen_sources = []

            # Depending on whether the gid is E or I, it'll form an EE or EI
            # connection. However, since all E* connections use the same
            # parameters, I don't need to check for a special case like I must
            # in the next conditional
            if 'Den_ex' in elms and elms['Den_ex'] > 0:
                sources = []
                # Can only connect with excitatory axons of E neurons
                for asource in self.neuronsE:
                    tid = asource
                    telms = synelms[asource]
                    if 'Axon_ex' in telms and telms['Axon_ex'] > 0:
                        # add the source multiple times, since it has multiple
                        # available contact points
                        sources.extend([tid]*int(telms['Axon_ex']))

                total_options_this_gid = len(sources)
                if (total_options_this_gid) > 0:
                    if self.syn_form_strategy == "random":
                        if (total_options_this_gid) > \
                                int(abs(elms['Den_ex'])):
                            chosen_sources = random.sample(
                                sources, int(abs(elms['Den_ex'])))
                        else:
                            chosen_sources = sources
                    elif self.syn_form_strategy == "distance":
                        chosen_sources = self.__get_nearest_ps_gaussian(
                            gid, sources, int(abs(elms['Den_ex'])),
                            w_mul=self.w_mul_E, max_p=self.max_p_E)

                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_sources),
                            total_options_this_gid, gid))

                    for cho in chosen_sources:
                        synelms[cho]['Axon_ex'] -= 1
                        syn_new_this_gid += 1
                        syn_dict = self.synDictEE.copy()
                        syn_dict['weight'] = random.gauss(
                            self.weightEE, self.weightSD_EE
                        )
                        nest.Connect([cho], [gid],
                                     conn_spec='one_to_one',
                                     syn_spec=syn_dict)

            # depending on whether the gid is E or I it'll form either an II
            # connection or an IE connection
            elif 'Den_in' in elms and elms['Den_in'] > 0:
                sources = []
                # can only connect to inhibitory axons of I neurons
                for asource in self.neuronsI:
                    tid = asource
                    telms = synelms[asource]
                    if 'Axon_in' in telms and telms['Axon_in'] > 0:
                        # add the source multiple times, since it has multiple
                        # available contact points
                        sources.extend([tid]*int(telms['Axon_in']))

                total_options_this_gid = len(sources)
                if (total_options_this_gid) > 0:
                    if self.syn_form_strategy == "random":
                        if (total_options_this_gid) > \
                                int(abs(elms['Den_in'])):
                            chosen_sources = random.sample(
                                sources, int(abs(elms['Den_in'])))
                        else:
                            chosen_sources = sources
                    elif self.syn_form_strategy == "distance":
                        chosen_sources = self.__get_nearest_ps_gaussian(
                            gid, sources, int(abs(elms['Den_in'])),
                            w_mul=self.w_mul_I, max_p=self.max_p_I)

                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_sources),
                            total_options_this_gid, gid))

                    for cho in chosen_sources:
                        synelms[cho]['Axon_in'] -= 1
                        syn_new_this_gid += 1
                        syn_dict = {}
                        # IE connection
                        if gid in self.neuronsE:
                            syn_dict = self.synDictIE.copy()
                            weight = self.weightIE
                            syn_dict['weight'] = weight

                            # also ensure that wmax is negative
                            wmax = random.gauss(self.weightIEmax,
                                                self.weightSD_IEmax)
                            while wmax >= 0.:
                                wmax = random.gauss(self.weightIEmax,
                                                    self.weightSD_IEmax)
                            syn_dict['Wmax'] = wmax
                        # II
                        else:
                            syn_dict = self.synDictII.copy()
                            syn_dict['weight'] = random.gauss(
                                self.weightII, self.weightSD_II
                            )
                        nest.Connect([cho], [gid],
                                     conn_spec='one_to_one',
                                     syn_spec=syn_dict)

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

    def __create_new_connections_from_pre(self, synelms):
        """Create new connections from the pre-synaptic side."""
        logging.debug("Rank {}: Creating using the {} strategy".format(
            self.rank, self.syn_form_strategy))
        syn_new_lpz_c_E = 0
        syn_new_lpz_b_E = 0
        syn_new_p_lpz_E = 0
        syn_new_o_E = 0
        syn_new_lpz_c_I = 0
        syn_new_lpz_b_I = 0
        syn_new_p_lpz_I = 0
        syn_new_o_I = 0
        current_sim_time = (str(nest.GetKernelStatus()['time']))

        for nrn in self.shuffled_neurons:
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
                    if self.syn_form_strategy == "random":
                        if (total_options_this_gid) > \
                                int(abs(elms['Axon_ex'])):
                            chosen_targets = random.sample(
                                (targetsE + targetsI),
                                int(abs(elms['Axon_ex'])))
                        else:
                            chosen_targets = (targetsE + targetsI)
                    elif self.syn_form_strategy == "distance":
                        chosen_targets = self.__get_nearest_ps_gaussian(
                            gid, (targetsE + targetsI),
                            int(abs(elms['Axon_ex'])), w_mul=self.w_mul_E,
                            max_p=self.max_p_E)

                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_targets),
                            total_options_this_gid, gid))

                    for cho in chosen_targets:
                        synelms[cho]['Den_ex'] -= 1
                        syn_new_this_gid += 1
                        syn_dict = {}
                        if cho in targetsE:
                            syn_dict = self.synDictEE.copy()
                            syn_dict['weight'] = random.gauss(
                                self.weightEE, self.weightSD_EE
                            )
                        else:
                            syn_dict = self.synDictEI.copy()
                            syn_dict['weight'] = random.gauss(
                                self.weightEI, (0.2 * self.weightEI)
                            )
                        nest.Connect([gid], [cho],
                                     conn_spec='one_to_one',
                                     syn_spec=syn_dict)

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
                    if self.syn_form_strategy == "random":
                        if (total_options_this_gid) > \
                                int(abs(elms['Axon_in'])):
                            chosen_targets = random.sample(
                                (targetsE + targetsI),
                                int(abs(elms['Axon_in'])))
                        else:
                            chosen_targets = (targetsE + targetsI)
                    elif self.syn_form_strategy == "distance":
                        chosen_targets = self.__get_nearest_ps_gaussian(
                            gid, (targetsE + targetsI),
                            int(abs(elms['Axon_in'])), w_mul=self.w_mul_I,
                            max_p=self.max_p_I)

                    logging.debug(
                        "Rank {}: {}/{} options chosen for neuron {}".format(
                            self.rank, len(chosen_targets),
                            total_options_this_gid, gid))

                    for target in chosen_targets:
                        syn_new_this_gid += 1
                        synelms[target]['Den_in'] -= 1
                        syn_dict = {}
                        if target in targetsE:
                            syn_dict = self.synDictIE.copy()

                            # ensure new weight is negative
                            weight = self.weightIE
                            syn_dict['weight'] = weight

                            # also ensure that wmax is negative
                            wmax = random.gauss(self.weightIEmax,
                                                self.weightSD_IEmax)
                            while wmax >= 0.:
                                wmax = random.gauss(self.weightIEmax,
                                                    self.weightSD_IEmax)
                            syn_dict['Wmax'] = wmax

                        else:
                            syn_dict = self.synDictII.copy()
                            syn_dict['weight'] = random.gauss(
                                self.weightII,
                                self.weightSD_II
                            )
                        nest.Connect([gid], [target],
                                     conn_spec='one_to_one',
                                     syn_spec=syn_dict)

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

        ca_fn_o_E = ("02-calcium-o_E-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(ca_fn_o_E, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.o_neurons_E) if
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

        ca_fn_o_I = ("02-calcium-o_I-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(ca_fn_o_I, 'w') as f:
            print("gid\tCa_conc", file=f, flush=True)
            allinfo = [[stat['global_id'], stat['Ca']] for
                       stat in nest.GetStatus(self.o_neurons_I) if
                       stat['local']]
            for info in allinfo:
                print("{}\t{}".format(info[0], info[1]), file=f)

    def __dump_syn_connections(self):
        """Dump pre-post syn pairs for existing synapses."""
        current_sim_time = (str(nest.GetKernelStatus()['time']))

        if current_sim_time != "0.0" and not self.is_str_p_enabled:
            logging.info("Not initial dump, structural plasticity not enabled")
            logging.info("Not dumping synapses - no change yet")
            return

        syn_fn_EE = ("08-syn_conns-EE-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(syn_fn_EE, 'w') as f:
            print("src\tdest\tweight", file=f, flush=True)
            conns = nest.GetConnections(source=self.neuronsE,
                                        target=self.neuronsE)
            weights = nest.GetStatus(conns, 'weight')
            for i in range(0, len(conns)):
                print("{}\t{}\t{}".format(
                    conns[i][0], conns[i][1], weights[i]
                ), file=f)

        syn_fn_EI = ("08-syn_conns-EI-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(syn_fn_EI, 'w') as f:
            print("src\tdest\tweight", file=f, flush=True)
            conns = nest.GetConnections(source=self.neuronsE,
                                        target=self.neuronsI)
            weights = nest.GetStatus(conns, 'weight')
            for i in range(0, len(conns)):
                print("{}\t{}\t{}".format(
                    conns[i][0], conns[i][1], weights[i]
                ), file=f)

        syn_fn_II = ("08-syn_conns-II-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(syn_fn_II, 'w') as f:
            print("src\tdest\tweight", file=f, flush=True)
            conns = nest.GetConnections(source=self.neuronsI,
                                        target=self.neuronsI)
            weights = nest.GetStatus(conns, 'weight')
            for i in range(0, len(conns)):
                print("{}\t{}\t{}".format(
                    conns[i][0], conns[i][1], weights[i]
                ), file=f)

        syn_fn_IE = ("08-syn_conns-IE-" +
                     str(self.rank) + "-" + current_sim_time +
                     ".txt")
        with open(syn_fn_IE, 'w') as f:
            print("src\tdest\tweight", file=f, flush=True)
            conns = nest.GetConnections(source=self.neuronsI,
                                        target=self.neuronsE)
            weights = nest.GetStatus(conns, 'weight')
            for i in range(0, len(conns)):
                print("{}\t{}\t{}".format(
                    conns[i][0], conns[i][1], weights[i]
                ), file=f)

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

            se_fn_o_E = (
                "05-se-o_E-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_o_E, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.o_neurons_E) if
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

            se_fn_o_I = (
                "05-se-o_I-" + str(self.rank) + "-" + current_sim_time
                + ".txt"
            )
            with open(se_fn_o_I, 'w') as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    "gid",
                    "axons", "axons_conn",
                    "dendrites_ex", "dendrites_ex_conn",
                    "dendrites_in", "dendrites_in_conn"),
                    file=f)
                allinfo = [[stat['global_id'], stat['synaptic_elements']] for
                           stat in nest.GetStatus(self.o_neurons_I) if
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

    def __get_neurons_from_region(self, num_neurons, first_point, last_point):
        """Get neurons in the centre of the grid.

        This will be used to get neurons for deaff, and also to get neurons for
        the centred pattern.
        """
        mid_point = [(x + y)/2 for x, y in zip(last_point, first_point)]
        neurons = self.location_tree.query(
            mid_point, k=num_neurons)[1]
        logging.debug("Rank {}: Got {}/{} neurons".format(
            self.rank, len(neurons), num_neurons))
        return neurons

    def __strengthen_pattern_connections(self, pattern_neurons):
        """Strengthen connections that make up the pattern."""
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        nest.SetStatus(connections, {"weight": self.weightPatternEE})
        logging.debug(
            "Rank {}: Number of connections strengthened: {}".format(
                self.rank, len(connections)))

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

    def update_connectivity(self):
        """Our implementation of structural plasticity."""
        if not self.is_rewiring_enabled:
            return
        logging.debug(
            "Rank {}: STRUCTURAL PLASTICITY: Updating connectivity".format(
                self.rank
            ))
        # shuffle before making this round of updates
        random.Random(23).shuffle(self.shuffled_neurons)

        syn_elms = self.__get_syn_elms()
        self.__delete_connections_from_pre(syn_elms)

        syn_elms_1 = self.__get_syn_elms()
        self.__delete_connections_from_post(syn_elms_1)

        syn_elms_2 = self.__get_syn_elms()
        self.__create_new_connections_from_pre(syn_elms_2)

        syn_elms_3 = self.__get_syn_elms()
        self.__create_new_connections_from_post(syn_elms_3)
        logging.debug(
            "Rank {}: STRUCTURAL PLASTICITY: Connectivity updated".format(
                self.rank
            ))

    def set_stability_threshold_I(self):
        """
        Sets the stability threshold for inhibitory synapses.
        """
        #  conns = nest.GetConnections(target=self.neuronsE,
        #  source=self.neuronsI)
        #  weightsIE = nest.GetStatus(conns, "weight")
        #  mean = abs(numpy.mean(weightsIE))
        #  std = abs(numpy.std(weightsIE))
        self.stability_threshold_I = abs(self.weightII +
                                         (2 * self.weightSD_II))
        self.stability_threshold_E = abs(self.weightEE +
                                         (2 * self.weightSD_EE))

    def update_mean_conductances(self):
        """
        Update mean conductances after stabilisation by synaptic plasticity.

        Only IE weights need an update currently, since all the others are
        static.
        """
        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsI)
        weightsIE = nest.GetStatus(conns, "weight")
        # these should already be negative
        self.weightIE = numpy.mean(weightsIE)
        self.weightSD_IE = self.weightIE/5.
        logging.debug("Rank {}: Updated mean IE weights ({}, {})".format(
            self.rank, self.weightIE, self.weightSD_IE))

    def invoke_metaplasticity(self):
        """Update growth curve parameters."""
        if self.is_metaplasticity_enabled and self.is_str_p_enabled:
            self.__set_str_p_params()
        logging.debug(
            "Rank {}: META PLASTICITY: Growth curves updated".format(
                self.rank
            ))

    def store_pattern_off_centre(self, offset=[0., 0.], track=False):
        """Store a pattern in the centre of LPZ."""
        logging.debug(
            "Rank {}: SIMULATION: Storing pattern {} in centre of LPZ".format(
                self.rank, self.pattern_count + 1))
        # first E neuron
        first_point = numpy.array(self.location_tree.data[0])
        # last E neuron
        # I neurons are spread among the E neurons and therefore do not make it
        # to the extremeties
        last_point = numpy.array(
            self.location_tree.data[len(self.neuronsE) - 1])
        centre_point = numpy.array(offset) + (first_point + last_point)/2
        self.store_pattern_with_centre(centre_point,
                                       int(1.25 * self.populations['P']),
                                       track=True)

    def store_pattern_with_centre(self, centre_point, num_neurons,
                                  track=False):
        """Store a pattern by specifying area extent."""
        logging.debug(
            "Rank {}: SIMULATION: Storing pattern {} centred at: {}".format(
                self.rank, self.pattern_count + 1, centre_point))
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
            "Rank {}: Number of patterns stored: {}".format(
                self.rank, self.pattern_count))

    def store_random_pattern(self, track=False, percent_in_lpz=None):
        """Store a pattern in a randomly selected set of neurons.

        Number of neurons in the pattern is taken from the stored parameter.

        :track: do we track the pattern or not.
        :percent_in_lpz: percent of pattern neurons that fall in LPZ,
            if unspecified, pick randomly. Note that the number of neurons in
            the LPZ must be kept in mind---if the LPZ has 200 neurons, one
            cannot request a pattern with 300 neurons in the LPZ.
        :returns: Nothing

        """
        logging.info(
            "Rank {}: SIMULATION: Storing pattern {} randomly".format(
                self.rank, self.pattern_count + 1))
        self.pattern_count += 1
        # Set seed
        seed = 182
        # if specified, take that much from LPZ and rest from outside
        # No need to change anything in recall, this will automatically change
        # the proportion of recall neurons taken from the LPZ also.
        if percent_in_lpz:
            if (int(self.populations['P'] * percent_in_lpz) <=
                    len(self.lpz_neurons_E)):
                pattern_neurons = (list(
                    random.Random(seed).sample(
                        self.lpz_neurons_E,
                        int(self.populations['P'] * percent_in_lpz))
                ) + list(random.Random(seed).sample(
                    self.p_lpz_neurons_E + self.o_neurons_E,
                    int(self.populations['P'] * (1 - percent_in_lpz))
                )))
                # shuffle so that the neurons in and out of the LPZ are not in
                # a contigous list, otherwise when we set up the recall
                # stimulus, we only get neurons from one of these two groups
                random.Random(seed).shuffle(pattern_neurons)
            else:
                logging.critical(
                    "Requested {} pattern neurons in the LPZ \
                    but the LPZ only has {} neurons. EXITING".format(
                        int(self.populations['P'] * percent_in_lpz),
                        len(self.lpz_neurons_E))
                )
                sys.exit(-1)
        else:
            pattern_neurons = list(
                # Set a seed, ensure all ranks get the same neurons
                random.Random(seed).sample(
                    self.neuronsE, self.populations['P'])
            )
        self.comm.barrier()
        self.__strengthen_pattern_connections(pattern_neurons)
        if track:
            self.__track_pattern(pattern_neurons)
        self.comm.barrier()
        logging.info(
            "Rank {}: {} neurons selected in pattern: {}".format(
                self.rank, len(pattern_neurons), self.pattern_count))

    def setup_pattern_for_recall(self, pattern_number):
        """
        Set up a pattern for recall.

        Creates a new poisson generator and connects it to a recall subset of
        this pattern - the poisson stimulus will run for the set
        recall_duration from the invocation of this method.
        """
        self.comm.barrier()
        # set up external stimulus
        pattern_neurons = self.patterns[pattern_number - 1]
        # pattern neurons are randomly chosen, so we can simply choose a
        # contiguous set
        recall_neurons = pattern_neurons[-self.populations['R']:]
        if self.rank == 0:
            fn = "00-recall-neurons-{}.txt".format(pattern_number)
            with open(fn, 'w') as fh:
                print("gid\txcor\tycor", file=fh, flush=True)
                for neuron in recall_neurons:
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[neuron - 1][0],
                        self.location_tree.data[neuron - 1][1]),
                        file=fh)

        # Do not provide stimulus to recall neurons that fall in the LPZ
        active_recall_neurons = list(set(recall_neurons) -
                                     set(self.lpz_neurons_E))

        # set up spike detectors
        stim_time = nest.GetKernelStatus()['time']
        neuronDictStim = {'rate': 200.,
                          'origin': stim_time,
                          'start': 0., 'stop': self.recall_duration}
        stim = nest.Create('poisson_generator', 1,
                           neuronDictStim)

        nest.Connect(stim, active_recall_neurons,
                     conn_spec=self.connDictStim)

        logging.info(
            "Rank {}: {}/{} recall neurons set up for pattern: {}".format(
                self.rank, len(active_recall_neurons), len(recall_neurons),
                pattern_number))
        self.recall_neurons.append(recall_neurons)
        self.comm.barrier()

    def recall_last_pattern(self, time):
        """
        Only setup the last pattern.

        An extra helper method, since we'll be doing this most.
        """
        logging.info("Rank {}: SIMULATION: RECALLING LAST PATTERN".format(
            self.rank
        ))
        self.recall_pattern(time, self.pattern_count)

    def recall_pattern(self, time, pattern_number):
        """
        Recall a pattern.

        We disable plasticities while we do this.
        We cannot store and load snapshots in NEST, so this is the next best
        thing to ensure that the network is not affected by the recall.
        """
        self.setup_pattern_for_recall(pattern_number)
        self.disable_syn_p()
        self.disable_rewiring()
        self.run_sim_phase(
            time, label="Recalling pattern {}".format(pattern_number))
        self.enable_rewiring()
        self.enable_syn_p()

    def deaff_network(self):
        """Deaff a the network."""
        logging.info("Rank{}: SIMULATION: deaffing spatial network".format(
            self.rank
        ))
        for nrn in self.lpz_neurons_E:
            nest.DisconnectOneToOne(self.poissonExt[0], nrn,
                                    syn_spec={'model': 'static_synapse'})
        for nrn in self.lpz_neurons_I:
            nest.DisconnectOneToOne(self.poissonExt[0], nrn,
                                    syn_spec={'model': 'static_synapse'})
        logging.info("Rank {}:SIMULATION: Network deafferentated".format(
            self.rank
        ))

        # record when the network is deaffed in seconds
        self.deaffed_at = (nest.GetKernelStatus()['time']/1000.)

    def dump_data(self):
        """Master datadump function."""
        logging.debug("Rank {}: Printing data to files".format(self.rank))
        self.__dump_syn_connections()
        self.__dump_ca_concentration()
        self.__dump_synaptic_elements_per_neurons()

    def close_files(self):
        """Close all files when the simulation is finished."""
        logging.debug("Rank {}: Closing open files".format(self.rank))
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

    def enable_syn_p(self):
        """
        Enable synaptic plasticity in IE synapses.

        It is possible to toggle synaptic plasticity by changing the synapse
        model that we use between vogels_sprekeler and static, but that would
        require more work. Simply changing eta is a much easier way.
        """
        if self.is_syn_p_enabled:
            logging.warning(
                "Rank {}: Synaptic plasticity already enabled!".format(
                    self.rank)
            )
            return

        # Enable STDP for future IE connections
        self.synDictIE['eta'] = self.etaIE

        conns = nest.GetConnections(source=self.neuronsI, target=self.neuronsE)
        nest.SetStatus(conns, {'eta': self.etaIE})
        self.is_syn_p_enabled = True
        logging.info(
            "Rank {}: Synaptic plasticity enabled in IE synapses.".format(
                self.rank)
        )

    def disable_syn_p(self):
        """Disable synaptic plasticity in IE synapses."""
        if not self.is_syn_p_enabled:
            logging.warning(
                "Rank {}: Synaptic plasticity already disabled!".format(
                    self.rank)
            )
            return

        # Disable STDP for future IE connections
        self.synDictIE['eta'] = 0.

        conns = nest.GetConnections(source=self.neuronsI, target=self.neuronsE)
        nest.SetStatus(conns, {'eta': 0.})
        self.is_syn_p_enabled = False
        logging.info(
            "Rank {}: Synaptic plasticity disabled in IE synapses.".format(
                self.rank)
        )

    def set_lpz_percent(self, lpz_percent):
        """Set up the network for deaff."""
        self.lpz_percent = lpz_percent
        logging.info("LPZ percent set to {}".format(self.lpz_percent))

    def stabilise(self, stab_time=0., label="Stabilising"):
        """Stabilise network."""
        # use default if not mentioned
        if stab_time:
            stabilisation_time = stab_time
        else:
            stabilisation_time = self.default_stabilisation_time

        self.run_sim_phase(stabilisation_time, label)

    def run_sim_phase(self, sim_time=2000, label="Phase A"):
        """Run a simulation phase."""
        start_time = time.clock()
        # take the smaller of the two intervals
        current_sim_time = nest.GetKernelStatus()['time']
        logging.info("{}: {}s simtime : phase started".format(
            label, current_sim_time/1000))
        phase_time = 0

        # make sure we run for the smallest interval
        if self.is_rewiring_enabled:
            run_duration = math.gcd(int(self.sp_update_interval),
                                    int(self.recording_interval))
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
            nest.Simulate(run_duration*1000.)
            current_sim_time = nest.GetKernelStatus()['time']
            logging.debug("Simulation time: {} seconds".format(
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
        end_time = time.clock()
        logging.info("{}: {}s simtime: phase ended".format(
            label, current_sim_time/1000.))
        logging.info("{}: phase took {}s in realtime".format(
            label, (end_time - start_time)/60.))

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
                print("{}: {} nS".format("mean weightEE", self.weightEE),
                      file=pfile)
                print("{}: {} ns".format("weightPatternEE",
                                         self.weightPatternEE),
                      file=pfile)
                print("{}: {} nS".format("mean weightEI", self.weightEI),
                      file=pfile)
                print("{}: {} nS".format("mean weightII", self.weightII),
                      file=pfile)
                print("{}: {} nS".format("mean weightIE", self.weightIE),
                      file=pfile)
                print("{}: {} nS".format("weightExtE", self.weightExtE),
                      file=pfile)
                print("{}: {} nS".format("weightExtI", self.weightExtI),
                      file=pfile)
                print("{}: {}".format("sparsity", self.sparsity),
                      file=pfile)
                print("{}: {}".format("E_threshold",
                                      self.stability_threshold_E),
                      file=pfile)
                print("{}: {}".format("I_threshold",
                                      self.stability_threshold_I),
                      file=pfile)

                print("{}: {}".format("eta_ax_E", self.eta_ax_e),
                      file=pfile)
                print("{}: {}".format("eps_ax_E", self.eps_ax_e),
                      file=pfile)
                print("{}: {}".format("omega_ax_E", self.omega_ax_e),
                      file=pfile)
                print("{}: {}".format("nu_ax_E", self.nu_ax_e),
                      file=pfile)
                print("{}: {}".format("eta_den_E_e", self.eta_den_e_e),
                      file=pfile)
                print("{}: {}".format("eps_den_E_e", self.eps_den_e_e),
                      file=pfile)
                print("{}: {}".format("omega_den_E_e", self.omega_den_e_e),
                      file=pfile)
                print("{}: {}".format("nu_den_E_e", self.nu_den_e_e),
                      file=pfile)
                print("{}: {}".format("eta_den_E_i", self.eta_den_e_i),
                      file=pfile)
                print("{}: {}".format("eps_den_E_i", self.eps_den_e_i),
                      file=pfile)
                print("{}: {}".format("omega_den_E_i", self.omega_den_e_i),
                      file=pfile)
                print("{}: {}".format("nu_den_E_i", self.nu_den_e_i),
                      file=pfile)

                print("{}: {}".format("eta_ax_I", self.eta_ax_i),
                      file=pfile)
                print("{}: {}".format("eps_ax_I", self.eps_ax_i),
                      file=pfile)
                print("{}: {}".format("omega_ax_I", self.omega_ax_i),
                      file=pfile)
                print("{}: {}".format("nu_ax_I", self.nu_ax_i),
                      file=pfile)
                print("{}: {}".format("eta_den_I_e", self.eta_den_i_e),
                      file=pfile)
                print("{}: {}".format("eps_den_I_e", self.eps_den_i_e),
                      file=pfile)
                print("{}: {}".format("omega_den_I_e", self.omega_den_i_e),
                      file=pfile)
                print("{}: {}".format("nu_den_I_e", self.nu_den_i_e),
                      file=pfile)
                print("{}: {}".format("eta_den_I_i", self.eta_den_i_i),
                      file=pfile)
                print("{}: {}".format("eps_den_I_i", self.eps_den_i_i),
                      file=pfile)
                print("{}: {}".format("omega_den_I_i", self.omega_den_i_i),
                      file=pfile)
                print("{}: {}".format("nu_den_I_i", self.nu_den_i_i),
                      file=pfile)
                print("{}: {}".format("deaffed_at", self.deaffed_at),
                      file=pfile)

    def update_time_windows(self,
                            stabilisation_time=None,
                            sp_update_interval=None,
                            recording_interval=None):
        """Set up stabilisation time."""
        if stabilisation_time:
            self.default_stabilisation_time = int(stabilisation_time)
        if sp_update_interval:
            self.sp_update_interval = int(sp_update_interval)
        if recording_interval:
            self.recording_interval = int(recording_interval)

        if math.gcd(int(self.recording_interval),
                    int(self.sp_update_interval)) == 1:
            logging.warning(
                "Recording ({}) and SP interval({}) are not multiples".format(
                    self.recording_interval, self.sp_update_interval))
            logging.warning(
                "Simulation will run in 1 second chunks only")

    def set_connectivity_strategies(self, formation_strategy,
                                    deletion_strategy):
        """Set the connection strategies for simulation.

        :formation_strategy: strategy to use while forming new connections
        :deletion_strategy: strategy to use while deleting connections
        """
        self.syn_del_strategy = deletion_strategy
        self.syn_form_strategy = formation_strategy


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        format='%(asctime)-15s: %(levelname)s: %(lineno)d: %(message)s',
        level=logging.INFO)

    store_patterns = False
    deafferentate_network = True
    repair_network = True
    simulation = Sinha2016()
    logging.info("Rank {}: SIMULATION STARTED".format(simulation.rank))

    # simulation setup
    # Setup network to handle plasticities
    simulation.setup_plasticity(True, True)
    simulation.set_connectivity_strategies("distance", "weight")
    # set up deaff extent, and neuron sets
    simulation.set_lpz_percent(0.05)
    # set up neurons, connections, spike detectors, files
    simulation.update_time_windows(stabilisation_time=1500.,
                                   sp_update_interval=1.,
                                   recording_interval=300.)
    simulation.setup_simulation()

    # synaptic plasticity stabilisation
    simulation.stabilise(label="Initial stabilisation")

    # Pattern related simulation
    if store_patterns:
        simulation.store_random_pattern(True)

        # stabilise network after storing patterns
        simulation.stabilise(label="Pattern stabilisation")
    # Set homoeostatic structural plasticity parameters to whatever the network
    # has achieved now
    simulation.invoke_metaplasticity()
    simulation.set_stability_threshold_I()
    # Enable structural plasticity for repair #
    simulation.print_simulation_parameters()
    simulation.enable_rewiring()

    #  Stabilise with both plasticities active
    #  update time windows
    simulation.update_time_windows(stabilisation_time=500.,
                                   sp_update_interval=1.,
                                   recording_interval=100.)
    simulation.stabilise(label="Both syn and str p")

    if deafferentate_network:
        # Deaff network
        simulation.deaff_network()

    if repair_network:
        simulation.update_time_windows(stabilisation_time=15000.,
                                       sp_update_interval=1.,
                                       recording_interval=1000.)
        simulation.stabilise(label="Repair")

    if store_patterns:
        # recall stored and tracked pattern
        simulation.recall_pattern(50, 1)

    simulation.close_files()
    logging.info("Rank {}: SIMULATION FINISHED SUCCESSFULLY".format(
        simulation.rank))
