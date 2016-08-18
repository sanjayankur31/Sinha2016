#!/usr/bin/env python3
"""
Enter one line description here.

File:

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
import os
import numpy
import subprocess


class GridSearch:

    """Set up your simulations for a grid search.


    This will modify the source in a branch, make changes, commit
    and then you can set these commits up on the cluster.
    """

    def __init__(self):
        """Initialise."""
        self.source = "/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/src/Sinha2016.py"
        self.branch = "master"

    def usage(self):
        """Print usage."""
        print("Usage:", file=sys.stderr)
        print("python3 grid_search.py <branch>", file=sys.stderr)
        print("Branch MUST be specified.", file=sys.stderr)

    def setup(self, branch, range_dict):
        """Set it up."""
        self.increment = range_dict['increment']
        self.branch = branch

        if len(range_dict['EE']) == 1:
            self.EE_min = range_dict['EE'][0]
            self.EE_max = range_dict['EE'][0] + self.increment
        elif len(range_dict['EE']) == 2:
            self.EE_min = range_dict['EE'][0]
            self.EE_max = range_dict['EE'][1] + self.increment
        else:
            print("EE not found in dict. Exiting.", file=sys.stderr)
            return False

        if len(range_dict['EI']) == 1:
            self.EI_min = range_dict['EI'][0]
            self.EI_max = range_dict['EI'][0] + self.increment
        elif len(range_dict['EI']) == 2:
            self.EI_min = range_dict['EI'][0]
            self.EI_max = range_dict['EI'][1] + self.increment
        else:
            print("EI not found in dict. Exiting.", file=sys.stderr)
            return False

        if len(range_dict['II']) == 1:
            self.II_min = range_dict['II'][0]
            self.II_max = range_dict['II'][0] + self.increment
        elif len(range_dict['II']) == 2:
            self.II_min = range_dict['II'][0]
            self.II_max = range_dict['II'][1] + self.increment
        else:
            print("II not found in dict. Exiting.", file=sys.stderr)
            return False

        return True

    def run(self):
        """Run."""
        # checkout the branch
        git_args = "checkout " + self.branch
        subprocess.call(['git', git_args])

        for weightEE in numpy.arange(self.EE_min, self.EE_max, self.increment):
            for weightEI in numpy.arange(self.EI_min, self.EI_max, self.increment):
                for weightII in numpy.arange(self.II_min, self.II_max, self.increment):

                    sed_args_EE = ['sed', '-i',
                                "s/weightEE = .*$/weightEE = {}".format(weightEE),
                                self.source]
                    subprocess.call(sed_args_EE)

                    sed_args_EI = ['sed', '-i',
                                "s/weightEI = .*$/weightEI = {}".format(weightEI),
                                self.source]
                    subprocess.call(sed_args_EI)

                    sed_args_II = ['sed', '-i',
                                "s/weightII = .*$/weightII = {}".format(weightII),
                                self.source]
                    subprocess.call(sed_args_II)

        git_args = "add " + self.source
        subprocess.call(['git', git_args])

if __name__ == "__main__":
    search = GridSearch()
    if len(sys.argv) != 2:
        search.usage()
        sys.exit(-1)
    else:
        branch = sys.argv[1]
        # dictionary that holds the required grid ranges
        # specify min, max if want a grid search, else specify only one value
        setup_dict = {
            'EE': [5.],
            'EI': [0.5, 3.],
            'II': [-30.],
            'increment': 0.5
        }
        if search.setup(branch, setup_dict):
            search.run()
