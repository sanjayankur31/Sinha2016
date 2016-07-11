#!/usr/bin/env python3
"""
Generate histograms to compare two sets of firing rates at a time.

File: dualHistogramPlotter.py

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

import sys
from textwrap import dedent
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob


class dualHistogramPlotter:

    """Generate histograms of two datasets in one figure."""

    def __init__(self):
        """Initialise."""
        self.filelist1 = ""
        self.filelist2 = ""
        self.neurons1 = 0
        self.neurons2 = 0

    def setup(self, set1, set2, num_neurons1, num_neurons2):
        """Setup things."""
        self.set1 = set1
        self.set2 = set2
        self.num_neurons1 = num_neurons1
        self.num_neurons2 = num_neurons2

        self.filelist1 = glob.glob("firing-rate-" + set1 + "-*.gdf")
        self.filelist2 = glob.glob("firing-rate-" + set2 + "-*.gdf")

        # sort them
        self.filelist1.sort()
        self.filelist2.sort()

        if not len(self.filelist1) == len(self.filelist2):
            print("{} {} files but {} {} files found".format(
                len(self.filelist1), set1,
                len(self.filelist2), set2) +
                "Please recheck the firing rate files")
            return False

        return True

    def run(self):
        """Main runner method to be used for command line invocation."""
        for i in range(0, len(self.filelist1)):
            with open(self.filelist1[i]) as f1:
                data1 = numpy.loadtxt(f1, delimiter='\t', dtype='float')
            with open(self.filelist2[i]) as f2:
                data2 = numpy.loadtxt(f2, delimiter='\t', dtype='float')

            self.plot_histogram(data1, data2, self.filelist1[i])

    def plot_histogram(self, data1, data2, filename):
        """Plot the histogram."""
        plt.figure(num=None, figsize=(16, 9), dpi=80)
        plt. xlabel("Firing rates")
        plt.ylabel("Number of neurons")

        plt.xticks(numpy.arange(0, 220, 20))
        plt.axis((0, 205, 0, 8000))
        plt.hist(data1, bins=100, alpha=0.5, label=self.set1)
        plt.hist(data2, bins=100, alpha=0.5, label=self.set2)
        time = (filename.split(sep='-')[3]).split('gdf')[0]
        plt.title("Histogram for " + self.set1 + " and " +
                  self.set2 + " at time " + time)
        output_filename = ("histogram-" + self.set1 + "-" + self.set2 + "-" +
                           time + "png")
        print("Storing {}".format(output_filename))
        plt.legend(loc="upper right")
        plt.savefig(output_filename)
        plt.close()

    def usage(self):
        """Print usage."""
        usage = ("Usage: \n" +
                 "python3 plot-histograms-time.py data1 data2" +
                 " num_neuron1 num_neurons2 " +
                 " time_start\n"
                 "python3 plot-histograms-time.py data1 data2" +
                 " num_neuron1 num_neurons2 " +
                 " time_start time_end")
        print(usage, file=sys.stderr)

if __name__ == "__main__":
    runner = dualHistogramPlotter()
    if len(sys.argv) == 6:
        runner.setup(sys.argv[1], sys.argv[2],
                     sys.argv[3], sys.argv[4])
        runner.run(sys.argv[5], sys.argv[5])
    elif len(sys.argv) == 7:
        runner.setup(sys.argv[1], sys.argv[2],
                     sys.argv[3], sys.argv[4])
        runner.run(sys.argv[5], sys.argv[6])
    else:
        print("Wrong arguments. Exiting.")
        runner.usage()
