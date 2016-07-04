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
import os
import gc


class getFiringRates:

    """Calculate firing rates for individual neurons."""

    def __init__(self):
        """Initialise."""
        self.spikes_filename = ""
        self.neurons_set_name = ""
        self.num_neurons = 0
        self.rows = 0.
        self.start = 0
        self.end = 0

    def setup(self, filename, neuron_set_name, num_neurons, rows=0.):
        """Setup."""
        self.spikes_filename = filename
        self.neurons_set_name = neuron_set_name
        self.num_neurons = int(num_neurons)
        self.rows = rows

        if not (
            os.path.exists(self.spikes_filename) and
            os.stat(self.spikes_filename).st_size > 0
        ):
            print("{} not found. Skipping.".format(self.spikes_filename),
                  file=sys.stderr)
            return False

        return True

    def __validate_input(self, dataframe):
        """Check to see the input file is a two column file."""
        if dataframe.shape[1] != 2:
            print("Data seems incorrect - should have 2 columns. " +
                  "Please check and re-run", file=sys.stderr)
            return False
        else:
            print("Read " + str(dataframe.shape[0]) +
                  " rows.")
            return True

    def print_firing_rates(self, time):
        """Print the file."""
        output_filename = "firing-rate-{}-{}.gdf".format(
            self.neurons_set_name, time)
        print("Printing firing rate values to {}".format(
            output_filename))

        output_file = open(output_filename, 'w')
        for rate in self.rates:
            print(rate, file=output_file)
        output_file.close()

    def run(self, timelist):
        """Main runner method."""
        # remember to convert to ms!
        sorted_timelist = numpy.sort(timelist)

        current = 0
        old_spikes = numpy.array([])
        old_times = numpy.array([])

        print("Reading spikes file {}".format(self.spikes_filename))
        for chunk in pandas.read_csv(self.spikes_filename, sep='\s+',
                                     names=["neuronID",
                                            "spike_time"],
                                     dtype={'neuronID': numpy.uint16,
                                            'spike_time': float},
                                     lineterminator="\n",
                                     skipinitialspace=True,
                                     header=None, index_col=None,
                                     chunksize=self.rows):

            if not self.__validate_input(chunk):
                print("Error in file. Skipping.", file=sys.stderr)
                return False

            # Only if you find the item do you print, else you read the next
            # chunk. Now, if all chunks are read and the item wasn't found, the
            # next items cannot be in the file either, since we're sorting the
            # file
            spikes = numpy.array(chunk.values[:, 0])
            times = numpy.array(chunk.values[:, 1])

            # 200 spikes per second = 2 spikes per 0.01 second (dt) per neuron
            # this implies 2 * 10000 spikes for 10000 neurons need to be kept
            # to make sure I have a proper sliding window of chunks
            if len(old_spikes) > 0:
                spikes = numpy.append(old_spikes, spikes)
                times = numpy.append(old_times, times)

            print(
                "Times from {} to {} being analysed containing {} rows".format(
                    times[0], times[-1], len(times)))

            while True:
                time = sorted_timelist[current]
                print("Looking for {}.".format(time))
                time *= 1000.

                # Find our values
                self.start = numpy.searchsorted(times,
                                                time - 1000.,
                                                side='left')
                self.end = numpy.searchsorted(times,
                                              time,
                                              side='right')
                # Not found at all, don't process anything
                if self.start == len(times):
                    print("Neurons not found, reading next chunk.")
                    break
                elif self.start < len(times) and self.end == len(times):
                    print("Found a boundary - reading another chunk.")
                    break
                else:
                    self.neurons = spikes[self.start:self.end]
                    counts = dict(collections.Counter(self.neurons))
                    self.rates = list(counts.values())

                    missing_neurons = self.num_neurons - len(self.rates)

                    print("Neurons found: {}".format(len(self.rates)))

                    # Add missing entries - affects the mean and std
                    # calculations. They have to be the right number
                    for entries in range(0, missing_neurons):
                        self.rates.append(0)

                    print("Neurons after appending zeros: {}".format(
                        len(self.rates)))

                    self.print_firing_rates(sorted_timelist[current])

                    current += 1
                    if current == len(sorted_timelist):
                        break

            if self.start < len(times):
                old_times = numpy.array(times[(self.start - len(times)):])
                old_spikes = numpy.array(spikes[(self.start - len(spikes)):])

            del spikes
            del times
            gc.collect()

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
