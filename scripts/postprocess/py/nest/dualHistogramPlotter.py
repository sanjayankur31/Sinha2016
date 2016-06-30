#!/usr/bin/env python3
"""
Generate histogram plotting files.

File: plot-histograms.py

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
import matplotlib.pylot as plt
from nest-getFiringRates import getFiringRates as gfr


class plotHistogram:

    """Generate histograms."""

    def __init__(self):
        """Initialise."""
        self.rateGetter1 = gfr()
        self.rateGetter2 = gfr()
        self.data1 = ""
        self.data2 = ""
        self.file_name1 = ""
        self.file_name2 = ""
        self.neurons1 = 0
        self.neurons2 = 0

    def set_data(self, data1, data2):
        """Set data names."""
        self.data1 = data1
        self.data2 = data2
        self.file_name1 = "spikes-{}.gdf".format(data1)
        self.file_name2 = "spikes-{}.gdf".format(data2)

    def set_neurons(self, neurons1, neurons2):
        """Set neuron numbers."""
        self.neurons1 = neurons1
        self.neurons2 = neurons2

    def run(self, data1, data2, time_start, time_end):
        """Main runner method."""
        time_start = float(time_start)
        time_end = float(time_end)

        if time_start == time_end:
            self._plot_histogram(data1, data2, time_start)
        else:
            for time in numpy.arange(time_start + 1., time_end + 1., 1.0,
                                     dtype=float):
                self._plot_histogram(data1, data2, time)

    def __plot_histogram(self, data1, data2, time)

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 plot-histograms-time.py data1 data2" +
                 " num_neuron1 num_neurons2 " +
                 " time_start time_end")
        print(usage, file=sys.stderr)

if __name__ == "__main__":
    runner = plotHistogram()
    if len(sys.argv) < 4:
        print("Wrong arguments. Exiting.")
        runner.usage()
    elif len(sys.argv) == 4:
        runner.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[3])
    elif len(sys.argv) == 5:
        runner.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Wrong arguments. Exiting.")
        runner.usage()
