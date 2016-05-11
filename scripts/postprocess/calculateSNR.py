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
import sys
import math
import collections


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
        self.recall_time = 0.

        self.num_neurons_pattern = 0
        self.num_neurons_noise = 0

    def setup(self, pattern_spikes_filename, noise_spikes_filename,
              num_neurons_pattern=800.,
              num_neurons_noise=7200., recall_time=0.):
        """Setup various things.

        Since we already have different files for pattern and noise spikes,
        I don't need to load the neuron lists for them and use them to come up
        with a firing rate - NEST already gives me different files
        for pattern, and different for noise.
        """
        self.pattern_spikes_filename = pattern_spikes_filename
        self.noise_spikes_filename = noise_spikes_filename

        self.num_neurons_pattern = int(num_neurons_pattern)
        self.num_neurons_noise = int(num_neurons_noise)
        self.recall_time = float(recall_time)

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

        # If anything isn't OK, we error out
        return (
            self.__validate_spike_input(self.spikes_pattern) and
            self.__validate_spike_input(self.spikes_noise)
        )

    def __validate_spike_input(self, spikes):
        """Check to see the input file is a two column file."""
        if spikes.shape[1] != 2:
            print("Pattern seems incorrect - should have 2 columns. " +
                  "Please check and re-run", file=sys.stderr)
            return False
        else:
            print("Read " + str(spikes.shape[0]) +
                  " rows.")
            return True

    def get_spikes(self, all_spikes):
        """Get spikes for this period."""
        times = all_spikes[:, 1]
        start = numpy.searchsorted(times,
                                   self.recall_time,
                                   side='left')
        end = numpy.searchsorted(times,
                                 self.recall_time + 1000.,
                                 side='right')
        return all_spikes[start:end, 0]

    def get_firing_rates(self, neurons, num_neurons):
        """Get firing rates for the specified neurons."""
        counts = dict(collections.Counter(neurons))
        rates = list(counts.values())

        missing_neurons = num_neurons - len(rates)

        print("Neurons found: {}".format(len(rates)))

        # Add missing entries - affects the mean and std calculations. They
        # have to be the right number
        for entries in range(0, missing_neurons):
            rates.append(0)

        print("Neurons after appending zeros: {}".format(len(rates)))
        return rates

    def calculate_snr(self):
        """Calculate the SNR."""
        spikes_pattern = self.get_spikes(self.spikes_pattern)
        # print("Pattern spikes: {}".format(spikes_pattern[0:]))
        spikes_noise = self.get_spikes(self.spikes_noise)
        # print("Noise spikes: {}".format(spikes_noise[0:]))

        rates_pattern = self.get_firing_rates(spikes_pattern,
                                              self.num_neurons_pattern)
        # print to file - for histograms
        output_file = open("recall-firing-rate-pattern.gdf", 'w')
        for rate in rates_pattern:
            print(rate, file=output_file)
        output_file.close()

        rates_noise = self.get_firing_rates(spikes_noise,
                                            self.num_neurons_noise)
        # print to file - for histograms
        output_file = open("recall-firing-rate-noise.gdf", 'w')
        for rate in rates_noise:
            print(rate, file=output_file)
        output_file.close()

        mean_pattern = numpy.mean(rates_pattern, dtype=float)
        print("Mean pattern is: {}".format(mean_pattern))
        mean_noise = numpy.mean(rates_noise, dtype=float)
        print("Mean noise is: {}".format(mean_noise))

        std_pattern = numpy.std(rates_pattern, dtype=float)
        print("STD pattern is: {}".format(std_pattern))
        std_noise = numpy.std(rates_noise, dtype=float)
        print("STD noise is: {}".format(std_noise))

        snr = 2. * (math.pow((mean_pattern - mean_noise),
                             2.)/(math.pow(std_pattern, 2.) +
                                  math.pow(std_noise, 2.)))

        print("SNR is: {}".format(snr))

        # Open the output file
        output_file = open("recall-snr.gdf", 'w')
        print("{}\t{}\t{}\t{}\t{}".format(mean_pattern, std_pattern,
                                          mean_noise, std_noise, snr),
              file=output_file)
        output_file.close()

        print("Result:\t{}\t{}\t{}\t{}\t{}".format(mean_pattern, std_pattern,
                                                   mean_noise, std_noise, snr))

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 calculateSNR.py" +
                 "pattern_spikes_filename noise_spikes_filename" +
                 "num_neurons_pattern" +
                 "num_neurons_noise recall_time"
                 )
        print(usage, file=stderr)

if __name__ == "__main__":
    converter = calculateSNR()
    if len(sys.argv) == 6:
        valid = converter.setup(str(sys.argv[1]), str(sys.argv[2]),
                                int(sys.argv[3]), int(sys.argv[4]),
                                int(sys.argv[5]))
        if valid:
            print("Processing ...")
            converter.calculate_snr()
            print("Finished ...")
    else:
        print("Incorrect arguments.", file=sys.stderr)
        converter.usage()
