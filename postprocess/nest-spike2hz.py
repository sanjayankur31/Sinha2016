#!/usr/bin/env python3
"""
Take a nest gdf file with spike times and calculate firing rates.

File: nest-spike2hz.py

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


class spike2hz:

    """Main class for utlity.

    Nest gdf file format:

        <neuron gid>    <spike_time>
    """

    def __init__(self):
        """Main init method."""
        self.input_file_name = ""
        self.output_file_name = ""

        # Initial indices
        self.left = 0.
        self.right = 0.
        self.dt = 1.  # ms
        self.num_neurons = 8000.

    def setup(self, input_file, output_file, num_neurons=8000.):
        """Setup various things."""
        self.input_file_name = input_file
        self.output_file_name = output_file

        self.spikes = numpy.loadtxt(fname=self.input_file_name, dtype=float)
        self.output_file = open(self.output_file_name, 'w')

        self.num_neurons = int(num_neurons)

        self.__validate_input()

    def __validate_input(self):
        """Check to see the input file is a two column file."""
        if self.spikes.shape[1] != 2:
            print("Data seems incorrect - should have 2 columns. " +
                  "Please check and re-run", file=sys.stderr)
        else:
            print("Read " + str(self.spikes.shape[0]) +
                  " rows.")

    def run(self):
        """Do the work."""
        self.times = self.spikes[:, 1]
        current_time = 0.
        while current_time <= math.ceil(self.times[-1] - 1000.):
            self.left += numpy.searchsorted(self.times[self.left:],
                                            current_time,
                                            side='left')
            self.right += numpy.searchsorted(self.times[self.right:],
                                             (current_time + 1000.),
                                             side='right')

            statement = ("{}\t{}\n".format(
                (current_time + 1000.)/1000.,
                (len(self.spikes[self.left:self.right, 0])/self.num_neurons)))

            self.output_file.write(statement)
            self.output_file.flush()

            current_time += self.dt

        self.output_file.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Incorrect arguments.", file=sys.stderr)
        print(file=sys.stderr)
        print("Usage: ", file=sys.stderr)
        print("\t" + str(sys.argv[0]) +
              " input_file output_file", file=sys.stderr)
    else:
        converter = spike2hz()
        if len(sys.argv) == 4:
            converter.setup(str(sys.argv[1]), str(sys.argv[2]),
                            int(sys.argv[3]))
        else:
            converter.setup(str(sys.argv[1]), str(sys.argv[2]))
        print("Processing ...")
        converter.run()
        print("Finished ...")
