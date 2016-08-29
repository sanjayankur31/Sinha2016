#!/usr/bin/env python3
"""
Plot raster of two sets of neurons.

File: dualRasterPlotter.py

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
import pandas
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys


class dualRasterPlotter:

    """Plot raster of two sets of neurons."""

    def __init__(self):
        """Initialise."""
        self.filename1 = ""
        self.filename2 = ""
        self.neurons1 = 0
        self.neurons2 = 0

    def setup(self, set1, set2, num_neurons1, num_neurons2,
              rows):
        """Setup things."""
        self.set1 = set1
        self.set2 = set2
        self.num_neurons1 = int(num_neurons1)
        self.num_neurons2 = int(num_neurons2)
        self.rows = int(rows)

        self.filename1 = "spikes-" + set1 + ".gdf"
        self.filename2 = "spikes-" + set2 + ".gdf"

        return True

    def run(self, timelist):
        """Main runner method to be used for command line invocation."""
        sorted_timelist = numpy.sort(timelist)

        self.print_spikes(sorted_timelist, self.filename1, self.set1)
        self.print_spikes(sorted_timelist, self.filename2, self.set2)

        self.plot_rasters(sorted_timelist)

    def print_spikes(self, timelist, filename, setname):
        """Print spikes."""
        sorted_timelist = numpy.sort(timelist)
        current = 0
        old_spikes = numpy.array([])
        old_times = numpy.array([])

        print("Reading spikes file {}".format(filename))
        for chunk in pandas.read_csv(filename, sep='\s+',
                                     names=["neuronID",
                                            "spike_time"],
                                     dtype={'neuronID': numpy.uint16,
                                            'spike_time': float},
                                     lineterminator="\n",
                                     skipinitialspace=True,
                                     header=None, index_col=None,
                                     chunksize=self.rows):

            if current == len(timelist):
                print("Processed all time values. Done.")
                break

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
                print("Looking for #{} - {}.".format(current, time))
                output_filename = ("spikes-" + setname + "-" + str(time) +
                                   ".gdf")
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
                    neurons = spikes[self.start:self.end]
                    spiketimes = times[self.start:self.end]
                    print("Neurons and times found: {} {}".format(
                        len(neurons), len(spiketimes)))

                    with open(output_filename, mode='wt') as f:
                        for i in range(0, len(neurons)):
                            print("{}\t{}".format(
                                neurons[i], spiketimes[i]), file=f)

                    current += 1
                    if current == len(sorted_timelist):
                        break

            if self.start < len(times):
                old_times = numpy.array(times[(self.start - len(times)):])
                old_spikes = numpy.array(spikes[(self.start - len(spikes)):])

            del spikes
            del times
            gc.collect()

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

    def plot_rasters(self, timelist):
        """Plot the rater."""
        for time in timelist:
            matplotlib.rcParams.update({'font.size': 30})
            plt.figure(num=None, figsize=(32, 18), dpi=80)
            plt. xlabel("Neurons")
            plt.ylabel("Time (ms)")
            plt.xticks(numpy.arange(0, 10020, 1000))

            filename1 = ("spikes-" + self.set1 + "-" + str(time) + ".gdf")
            filename2 = ("spikes-" + self.set2 + "-" + str(time) + ".gdf")

            if not (
                os.path.exists(filename1) and
                os.stat(filename1).st_size > 0
            ):
                print("{} not found. Skipping.".format(filename1),
                      file=sys.stderr)
                return False

            if not (
                os.path.exists(filename2) and
                os.stat(filename2).st_size > 0
            ):
                print("{} not found. Skipping.".format(filename2),
                      file=sys.stderr)
                return False

            neurons1DF = pandas.read_csv(filename1, sep='\s+',
                                         lineterminator="\n",
                                         skipinitialspace=True,
                                         header=None, index_col=None)
            neurons1 = neurons1DF.values
            neurons2DF = pandas.read_csv(filename2, sep='\s+',
                                         lineterminator="\n",
                                         skipinitialspace=True,
                                         header=None, index_col=None)
            neurons2 = neurons2DF.values
            # Don't need to shift them - already numbered nicely

            plt.plot(neurons1[:, 0], neurons1[:, 1], ".", markersize=0.6,
                     label=self.set1)
            plt.plot(neurons2[:, 0], neurons2[:, 1], ".", markersize=0.6,
                     label=self.set2)

            plt.title("Raster for " + self.set1 + " and " +
                      self.set2 + " at time " + str(time))
            output_filename = ("raster-" + self.set1 + "-" + self.set2 + "-"
                               + str(time) + ".png")

            print("Storing {}".format(output_filename))
            plt.legend(loc="upper right")
            plt.savefig(output_filename)
            plt.close()

        return True

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
    runner = dualRasterPlotter()
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
