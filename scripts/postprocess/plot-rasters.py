#!/usr/bin/env python3
"""
Plot raster plots per half second.

File: plot-rasters.py

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

import textwrap
import sys
from numpy import linspace


class plotRasters:

    """Plot rasters per half second."""

    def __init__(self):
        """Initialise."""
        self.header = """
        set term pngcairo font "OpenSans, 28" size 1920,4096
        set xlabel "Time"
        set ylabel "Neurons"
        """
        self.input_file_I = "spikes-I.gdf"
        self.input_file_E = "spikes-E.gdf"

    def plot_all_rasters(self, interval_start, interval_end):
        """Plot all rasters in interval."""
        interval = interval_start

        while interval != interval_end:
            interval += 0.5
            self.plot_raster(interval)

    def plot_raster(self, interval):
        """Plot raster."""
        print("Generating file for {}, {}".format(interval - 0.5, interval))
        title_command = """
        set title "0.5 second interval ending at {}"
        """.format(interval)
        output_plot_file = "raster-{}-{}.png".format(interval - 0.5, interval)
        output_command = """set output "{}" """.format(output_plot_file)
        range_command = """set xrange[{}:{}]""".format(interval - 0.5,
                                                       interval)
        plot_command = (""" plot""" +
                        """ "<(sed -n '/^{}/,/^{}/p;/^{}/q' {})" """.format(
                            interval - 0.5, interval, interval,
                            self.input_file_I) +
                        """using 1:($2+8000) with points ps 0.5 lw 0.5""" +
                        """ title "", """ +
                        """ "<(sed -n '/^{}/,/^{}/p;/^{}/q' {})" """.format(
                            interval - 0.5, interval, interval,
                            self.input_file_E) +
                        """using 1:2 with points ps 0.5 lw 0.5 title "" """
                        )

        output_file = open("plot-raster-{}-{}.plt".format(interval - 0.5,
                                                          interval), 'w')

        print(textwrap.dedent(self.header), file=output_file)
        print(textwrap.dedent(output_command), file=output_file)
        print(textwrap.dedent(title_command), file=output_file)
        print(textwrap.dedent(range_command), file=output_file)
        print(textwrap.dedent(plot_command), file=output_file)

        output_file.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        plotter = plotRasters()
        plotter.plot_raster(float(sys.argv[1]))
    elif len(sys.argv) == 3:
        plotter = plotRasters()
        plotter.plot_all_rasters(float(sys.argv[1]), float(sys.argv[2]))
    else:
        print("I don't understand what you want.", file=sys.stderr)
