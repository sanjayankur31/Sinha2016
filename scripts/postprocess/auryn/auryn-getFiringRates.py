#!/usr/bin/env python3
"""
Calculate firing rates for individual neurons for a particular time.

File: getFiringRates.py

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


class getFiringRates:

    """Calculate firing rates for individual neurons."""

    def __init__(self):
        """Initialise."""
        self.spikes_filename = ""
        self.neurons_set_name = ""
        self.num_neurons = 0

    def __setup(self, filename, neuron_set_name, num_neurons):
        """Setup."""
        self.spikes_filename = filename
        self.neurons_set_name = neuron_set_name
        self.num_neurons = int(num_neurons)

    def __read_file(self):
        """Read the file."""
        print("Reading spikes file {}".format(self.spikes_filename))
        spikesDF = pandas.read_csv(self.spikes_filename,
                                   sep='\s+', dtype=float,
                                   lineterminator="\n",
                                   skipinitialspace=True, header=None,
                                   index_col=None, names=None)
        self.spikes = spikesDF.values

        return self.__validate_spike_input()

    def __validate_spike_input(self):
        """Check to see the input file is a two column file."""
        if self.spikes.shape[1] != 2:
            print("Pattern seems incorrect - should have 2 columns. " +
                  "Please check and re-run", file=sys.stderr)
            return False
        else:
            print("Read " + str(self.spikes.shape[0]) +
                  " rows.")
            return True

    def __get_spikes(self, time):
        """Get spikes for this period."""
        print("Getting spikes for {} - {}".format(time - 1., time))
        times = self.spikes[:, 0]
        start = numpy.searchsorted(times,
                                   time - 1.,
                                   side='left')
        end = numpy.searchsorted(times,
                                 time,
                                 side='right')
        self.neurons = self.spikes[start:end, 1]

    def __get_firing_rates(self):
        """Get firing rates for the specified neurons."""
        counts = dict(collections.Counter(self.neurons))
        self.rates = list(counts.values())

        missing_neurons = self.num_neurons - len(self.rates)

        print("Neurons found: {}".format(len(self.rates)))

        # Add missing entries - affects the mean and std calculations. They
        # have to be the right number
        for entries in range(0, missing_neurons):
            self.rates.append(0)

        print("Neurons after appending zeros: {}".format(len(self.rates)))

    def __print_firing_rates(self, time):
        """Print the file."""
        output_filename = "firing-rate-{}-{}.gdf".format(
            self.neurons_set_name, time)
        print("Printing firing rate values to {}".format(
            output_filename))

        output_file = open(output_filename, 'w')
        for rate in self.rates:
            print(rate, file=output_file)
        output_file.close()

    def run(self, filename, neuron_set_name, num_neurons,
            time_start, time_end):
        """Main runner method."""
        time_start = round(float(time_start))
        time_end = round(float(time_end))
        self.__setup(filename, neuron_set_name, num_neurons)
        if not self.__read_file():
            print("Files not valid. Exiting.", file=sys.stderr)
            return
        else:
            if time_start == time_end:
                self.__get_spikes(time_start)
                self.__get_firing_rates()
                self.__print_firing_rates(time_start)
            else:
                for time in numpy.arange(time_start + 1., time_end + 1., 1.0,
                                         dtype=float):
                    self.__get_spikes(time)
                    self.__get_firing_rates()
                    self.__print_firing_rates(time)

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 getFiringRates.py " +
                 "spike_file_name " +
                 "neuron_set_name " +
                 "num_neurons time\n" +
                 "python3 getFiringRates.py " +
                 "spike_file_name " +
                 "neuron_set_name num_neurons " +
                 "time_start time_end "
                 )
        print(usage, file=sys.stderr)

if __name__ == "__main__":
    runner = getFiringRates()
    if len(sys.argv) < 5:
        print("Wrong arguments. Exiting.")
        runner.usage()
    elif len(sys.argv) == 5:
        runner.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                   sys.argv[4])
    elif len(sys.argv) == 6:
        runner.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                   sys.argv[5])
    else:
        print("Wrong arguments. Exiting.")
        runner.usage()
