#!/usr/bin/env python
"""
Calculate SNR from spike files.

File: scripts/postprocess/calculateSnapshotStats.py

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


class calculateSnapshotStats:

    """
    Calculate SNR from spike files.

    This takes the data1 and data2 spike files, extracts the required
    spikes, does the maths, and prints the SNR into a file.
    """

    def __init__(self):
        """Initialise variables."""
        self.data1_spikes_filename = ""
        self.data2_spikes_filename = ""
        self.time = 0.

        self.num_neurons_data1 = 0
        self.num_neurons_data2 = 0

    def setup(self, data1, data2,
              num_neurons_data1=800.,
              num_neurons_data2=7200., time=0.):
        """Setup various things.

        Since we already have different files for data1 and data2
        spikes, I don't need to load the neuron lists for them and use them to
        come up with a firing rate - NEST already gives me different files for
        data1, and different for data2.
        """
        self.data1_spikes_filename = "spikes-{}.gdf".format(data1)
        self.data2_spikes_filename = "spikes-{}.gdf".format(data2)
        self.data1 = data1
        self.data2 = data2

        self.num_neurons_data1 = int(num_neurons_data1)
        self.num_neurons_data2 = int(num_neurons_data2)
        self.time = float(time)

        # Get spikes
        spikesDF_data1 = pandas.read_csv(self.data1_spikes_filename,
                                         sep='\s+', dtype=float,
                                         lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
        self.spikes_data1 = spikesDF_data1.values

        spikesDF_data2 = pandas.read_csv(self.data2_spikes_filename,
                                         sep='\s+', dtype=float,
                                         lineterminator="\n",
                                         skipinitialspace=True,
                                         header=None, index_col=None,
                                         names=None)
        self.spikes_data2 = spikesDF_data2.values

        # If anything isn't OK, we error out
        return (
            self.__validate_spike_input(self.spikes_data1) and
            self.__validate_spike_input(self.spikes_data2)
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
                                   self.time,
                                   side='left')
        end = numpy.searchsorted(times,
                                 self.time + 1000.,
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
        spikes_data1 = self.get_spikes(self.spikes_data1)
        # print("Pattern spikes: {}".format(spikes_data1[0:]))
        spikes_data2 = self.get_spikes(self.spikes_data2)
        # print("Background spikes: {}".format(spikes_data2[0:]))

        rates_data1 = self.get_firing_rates(spikes_data1,
                                            self.num_neurons_data1)
        # print to file - for histograms
        output_file = open("firing-rate-{}-{}.gdf".format(self.data1,
                                                          self.time), 'w')
        for rate in rates_data1:
            print(rate, file=output_file)
        output_file.close()

        rates_data2 = self.get_firing_rates(spikes_data2,
                                            self.num_neurons_data2)
        # print to file - for histograms
        output_file = open("firing-rate-{}-{}.gdf".format(self.data2,
                                                          self.time), 'w')
        for rate in rates_data2:
            print(rate, file=output_file)
        output_file.close()

        mean_data1 = numpy.mean(rates_data1, dtype=float)
        print("Mean data1 is: {}".format(mean_data1))
        mean_data2 = numpy.mean(rates_data2, dtype=float)
        print("Mean data2 is: {}".format(mean_data2))

        std_data1 = numpy.std(rates_data1, dtype=float)
        print("STD data1 is: {}".format(std_data1))
        std_data2 = numpy.std(rates_data2, dtype=float)
        print("STD data2 is: {}".format(std_data2))

        snr = 2. * (math.pow((mean_data1 - mean_data2),
                             2.)/(math.pow(std_data1, 2.) +
                                  math.pow(std_data2, 2.)))

        print("SNR is: {}".format(snr))

        # Open the output file
        output_file = open("snr-{}.gdf".format(self.time), 'w')
        print("{}\t{}\t{}\t{}\t{}".format(mean_data1, std_data1,
                                          mean_data2, std_data2,
                                          snr),
              file=output_file)
        output_file.close()

        print("Result:\t{}\t{}\t{}\t{}\t{}".format(mean_data1, std_data1,
                                                   mean_data2,
                                                   std_data2, snr))

    def plot_histogram(self):
        """Plot a histogram."""
        print("Generating file for {}".format(self.time))
        command = """
        reset
        max=200.
        min=0.
        n1=200
        width1=(max-min)/n1
        n2=200
        width2=(max-min)/n2
        hist(x,width)=width*floor(x/width)+width/2.0
        set term pngcairo font "OpenSans, 28" size 1920,1028
        set output "histogram-{}-{}-{}.png"
        set xrange[min:max]
        set yrange[0:]
        set offset graph 0.05,0.05,0.05,0.0
        set xtics min,20,max
        set boxwidth width*0.9
        set style fill transparent solid 0.5 #fillstyle
        set tics out nomirror
        set xlabel "Firing rate"
        set ylabel "Frequency"
        plot "{}-firing-rate.gdf" u (hist($1,width1)):(1.0) smooth freq w boxes title "{}", "{}-firing-rate.gdf" u (hist($1,width2)):(1.0) smooth freq w boxes title "{}";
        """.format(self.time, self.data1, self.data2, self.data1, self.data1,
                   self.data2, self.data2)

        output_file = open("plot-histogram-{}.plt".format(self.time), 'w')
        print(command, file=output_file)
        output_file.close()

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 calculateSnapshotStats.py" +
                 "data1 data2" +
                 "num_neurons_data1" +
                 "num_neurons_data2 time"
                 )
        print(usage, file=sys.stderr)

if __name__ == "__main__":
    converter = calculateSnapshotStats()
    if len(sys.argv) == 6:
        valid = converter.setup(str(sys.argv[1]), str(sys.argv[2]),
                                int(sys.argv[3]), int(sys.argv[4]),
                                int(sys.argv[5]))
        if valid:
            print("Processing ...")
            converter.calculate_snr()
            converter.plot_histogram()
            print("Finished ...")
    elif len(sys.argv) == 7:
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
