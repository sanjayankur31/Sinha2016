#!/usr/bin/env python3
"""
Combine weightsuminfo files from Auryn and print out the mean weight.

File: auryn-combine-weightsuminfo.py

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

import numpy
import sys
import glob
import math
import pandas


class getMeanWeights:

    """
    Combine weightsum files and calculate the mean weight.

    Only to be used for files generated using the WeightSumMonitor.
    """

    def __init__(self):
        """Main init method."""
        self.output_file_name = ""

        self.usage = ("Usage: \npython3 auryn-get-mean-weight.py " +
                      "output_file_name number_synapses")

        # Initial indices
        self.num_synapses = 0.

    def run(self, output_file, num_synapses=(8000. * 2000. * 0.02)):
        """Set up the script."""
        self.num_synapses = float(num_synapses)
        self.input_files = glob.glob("*_ie_stdp.weightinfo")
        weightsDF = pandas.DataFrame()

        for f in self.input_files:
            df1 = pandas.read_csv(f, sep=" ", header=None, usecols=[1],
                                  dtype=float)
            weightsDF = pandas.concat([weightsDF, df1], axis=1)

        print(weightsDF)
        sumDF = weightsDF.sum(axis=1)
        print(sumDF)

        print(sumDF.divide(self.num_synapses))

        # self.output_file = open(self.output_file_name, 'w')

    def print_usage(self):
        """Print usage."""
        print(self.usage, file=sys.stderr)


if __name__ == "__main__":
    converter = getMeanWeights()
    if len(sys.argv) == 3:
        converter.run(str(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Incorrect arguments.", file=sys.stderr)
        print(file=sys.stderr)
        converter.print_usage()
