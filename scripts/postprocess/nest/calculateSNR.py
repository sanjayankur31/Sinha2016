#!/usr/bin/env python3
"""
Calculate SNR from two sets of unlabelled firing rate files.

File: calculateSNR.py

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
import sys
import numpy


class calculateSNR:

    """Calculate SNR from two sets of unlabelled firing rate files."""

    def run(self, file1, file2):
        """Main runner method."""
        firing_rates1 = pandas.read_csv(file1,
                                        sep='\s+', dtype=float,
                                        lineterminator="\n",
                                        skipinitialspace=True, header=None,
                                        index_col=None, names=None)
        rates1 = firing_rates1.values
        firing_rates2 = pandas.read_csv(file2,
                                        sep='\s+', dtype=float,
                                        lineterminator="\n",
                                        skipinitialspace=True, header=None,
                                        index_col=None, names=None)
        rates2 = firing_rates2.values

        mean1 = numpy.mean(rates1, dtype=float)
        print("Mean1 is: {}".format(mean1))
        mean2 = numpy.mean(rates2, dtype=float)
        print("Mean2 is: {}".format(mean2))

        std1 = numpy.std(rates1, dtype=float)
        print("STD1is: {}".format(std2))
        std2 = numpy.std(rates2, dtype=float)
        print("STD2 is: {}".format(std2))

        snr = 2. * (math.pow((mean1 - mean2),
                             2.)/(math.pow(std1, 2.) +
                                  math.pow(std2, 2.)))

        print("SNR is: {}".format(snr))

    def usage(self):
        """Print usage."""
        usage = ("Usage: \npython3 calculateSNR.py " +
                 "file1 file2" +
                 )
        print(usage, file=sys.stderr)

if __name__ == "__main__":
    runner = calculateSNR()
    if len(sys.argv) != 3:
        print("Wrong arguments.", file=sys.stderr)
        runner.usage()
    else:
        runner.run(sys.argv[1], sys.argv[2])
