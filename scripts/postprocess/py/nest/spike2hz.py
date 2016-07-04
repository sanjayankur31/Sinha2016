#!/usr/bin/env python3
"""
Take a nest gdf file with spike times and calculate mean firing rates.

File: spike2hz.py

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

import numpy
import sys
import math
import pandas
import os.path
import gc


class spike2hz:

    """Main class for utlity.

    Nest gdf file format:

        <neuron gid>    <spike_time>

    Takes an entire spike file and generates the mean firing rate file to be
    used for time graphs.
    """

    def __init__(self):
        """Main init method."""
        self.input_filename = ""
        self.output_filename = ""
        self.usage = ("nest-spike2hz.py: generate mean firing rate file " +
                      "from spike file\n\n" +
                      "Usage: \npython3 nest-spike2hz.py " +
                      "input_filename output_filename number_neurons" +
                      " rows")

        # Initial indices
        self.left = 0.
        self.right = 0.
        self.dt = 1.  # ms
        self.num_neurons = 8000.
        self.rows = 0.

    def setup(self, input_filename, output_filename, num_neurons=8000.,
              rows=0.):
        """Setup various things."""
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.rows = rows
        self.output_file = open(self.output_filename, 'w')
        self.num_neurons = int(num_neurons)

        if not (
            os.path.exists(self.input_filename) and
            os.stat(self.input_filename).st_size > 0
        ):
            print("File not found. Skipping.", file=sys.stderr)
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

    def run(self):
        """Do the work."""
        start_row = 0
        current_time = 1000.
        old_spikes = numpy.array([])
        old_times = numpy.array([])
        for chunk in pandas.read_csv(self.input_filename, sep='\s+',
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
            print("Current time is {}".format(current_time))

            # Reset chunks
            self.left = 0
            self.right = 0

            while (current_time < math.floor(times[-1])):
                self.left += numpy.searchsorted(times[self.left:],
                                                (current_time - 1000.),
                                                side='left')
                self.right = self.left + numpy.searchsorted(
                    times[self.left:], current_time,
                    side='right')

                statement = ("{}\t{}\n".format(
                    current_time/1000.,
                    (
                        len(
                            spikes[self.left:self.right]
                        )/self.num_neurons)))

                self.output_file.write(statement)
                self.output_file.flush()

                current_time += self.dt

            print("Printed till {}".format(current_time))
            old_times = numpy.array(times[(self.left - len(times)):])
            old_spikes = numpy.array(spikes[(self.left - len(spikes)):])

            del spikes
            del times
            gc.collect()

        self.output_file.close()

    def print_usage(self):
        """Print usage."""
        print(self.usage, file=sys.stderr)

if __name__ == "__main__":
    converter = spike2hz()
    if len(sys.argv) == 3:
        converter.setup(str(sys.argv[1]), str(sys.argv[2]))
        print("Processing ...")
        converter.run()
        print("Finished ...")
    elif len(sys.argv) == 4:
        converter.setup(str(sys.argv[1]), str(sys.argv[2]),
                        int(sys.argv[3]))
        converter.run()
        print("Finished ...")
    elif len(sys.argv) == 5:
        converter.setup(str(sys.argv[1]), str(sys.argv[2]),
                        int(sys.argv[3]), int(sys.argv[4]))
        print("Processing ...")
        converter.run()
        print("Finished ...")
    else:
        print("Incorrect arguments.", file=sys.stderr)
        print(file=sys.stderr)
        converter.print_usage()
