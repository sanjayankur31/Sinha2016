#!/usr/bin/env python
"""
Calculate SNR from spike files.

File: scripts/postprocess/calculateSNR.py

Copyright 2016 Ankur Sinha
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

import pandas
import numpy


class calculateSNR:

    """
    Calculate SNR from spike files.

    This takes the pattern and noise spike files, extracts the required spikes,
    does the maths, and prints the SNR into a file.
    """

    def __init__(self):
        """Initialise variables."""
        self.pattern_spikes_filename = ""
        self.noise_spikes_filename = ""
        self.output_filename = ""
        self.recall_time = 0.

        self.num_neurons_pattern = 0
        self.num_neurons_noise = 0

        self.pattern_neurons = []
        self.noise_neurons = []

    def setup(self, pattern_spikes_filename, noise_spikes_filename,
              output_filename, num_neurons_pattern=800.,
              num_neurons_noise=7200., recall_time=0.):
        """Setup various things.

        Since we already have different files for pattern and noise spikes,
        I don't need to load the neuron lists for them and use them to come up
        with a firing rate - NEST already gives me different files
        for pattern, and different for noise.
        """
        self.pattern_spikes_filename = pattern_spikes_filename
        self.noise_spikes_filename = noise_spikes_filename
        self.output_filename = output_filename

        self.num_neurons_pattern = int(num_neurons_pattern)
        self.num_neurons_noise = int(num_neurons_noise)
        self.recall_time = int(recall_time)

        # Get spikes
        spikesDF_pattern = pandas.read_csv(self.pattern_spikes_filename,
                                           sep='\s+', dtype=float,
                                           lineterminator="\n",
                                           skipinitialspace=True, header=None,
                                           index_col=None, names=None)
        self.spikes_pattern = spikesDF_pattern.values

        spikesDF_noise = pandas.read_csv(self.noise_spikes_filename, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
        self.spikes_noise = spikesDF_noise.values

        # Open the output file
        self.output_file = open(self.output_filename, 'w')

        # If anything isn't OK, we error out
        return (
            self.__validate_spike_input(self.spikes_pattern) and
            self.__validate_spike_input(self.spikes_noise)
        )

    def __validate_spike_input(self, spikes):
        """Check to see the input file is a two column file."""
        if spikes[1] != 2:
            print("Pattern seems incorrect - should have 2 columns. " +
                  "Please check and re-run", file=sys.stderr)
            return False
        else:
            print("Read " + str(self.spikes.shape[0]) +
                  " rows.")
            return True

    def get_spikes(self, all_spikes):
        """Get spikes for this period."""
        times = all_spikes[:, 1]
        start = numpy.searchsorted(times,
                                   self.recall_time,
                                   side='left')
        end = numpy.searchsorted(times,
                                 self.recall_time,
                                 side='right')
        return all_spikes[start:end, 0]

    def get_firing_rates(self, neurons, num_neurons):
        """Get firing rates for the specified neurons."""
        rates = [0 for i in range(0, num_neurons)]
        for neuron in neurons:
            rates[neuron] += 1

        return rates

    def calculate_snr(self):
        """Calculate the SNR."""
        spikes_pattern = self.get_spikes(self.spikes_pattern)
        spikes_noise = self.get_spikes(self.spikes_noise)

        rates_pattern = self.get_firing_rates(spikes_pattern)
        rates_noise = self.get_firing_rates(spikes_noise)

        mean_pattern = numpy.mean(rates_pattern, dtype=float)
        mean_noise = numpy.mean(rates_noise, dtype=float)

        std_pattern = numpy.std(rates_pattern, dtype=float)
        std_noise = numpy.std(rates_noise, dtype=float)

        snr = 2. * (math.pow((mean_pattern - mean_noise),
                             2.)/(math.pow(std_pattern, 2.) +
                                  math.pow(std_noise, 2.)))

        return snr
