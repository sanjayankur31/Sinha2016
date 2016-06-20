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


class plotHistogram:

    """Generate histogram plotting files."""

    def print_plt_file(self, data1, data2, time):
        """Print the final file."""
        print("Generating file for {} and {} at {}".format(data1,
                                                           data2, time))

        command = """
        reset
        max=200.
        min=0.
        n=200
        width=(max-min)/n
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
        """.format(data1, data2, time)
        plot_command = (
            """set title "histogram for {} and {} at {}"\n""".format(
                data1, data2, time
            ) +
            """plot "firing-rate-{}-{}.gdf" """.format(data1, time) +
            """u (hist($1,width)):(1.0) smooth freq """ +
            """w boxes title "{}", """.format(data1) +
            """ "firing-rate-{}-{}.gdf" """.format(data2, time) +
            """u (hist($1,width)):(1.0) smooth """ +
            """freq w boxes title "{}" """.format(data2))

        output_file = open("plot-histogram-{}-{}-{}.plt".format(
            data1, data2, time), 'w')
        print(dedent(command), file=output_file)
        print(plot_command, file=output_file)
        output_file.close()

    def run(self, data1, data2, time_start, time_end):
        """Main runner method."""
        time_start = float(time_start)
        time_end = float(time_end)

        if time_start == time_end:
            self.print_plt_file(data1, data2, time_start)
        else:
            for time in numpy.arange(time_start + 1., time_end + 1., 1.0,
                                     dtype=float):
                self.print_plt_file(data1, data2, time)

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 plot-histograms-time.py data1 data2" +
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
