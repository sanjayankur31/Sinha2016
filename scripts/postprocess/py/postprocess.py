#!/usr/bin/env python3
"""
Main post processing method.

File: postprocess.py

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

from config import Config
import sys
import os


class Postprocess:

    """Main post process worker class."""

    def __init__(self, simulator):
        """Initialise."""
        self.simulator = simulator
        self.tasklist = "tasklist-{}.ini".format(simulator)

    def __load_config(self):
        """Load configuration file."""
        if os.path.isfile(self.tasklist):
            self.config = Config(self.tasklist)
            print("Config file {} loaded successfully.".format(self.tasklist))
        else:
            sys.exit("Could not find tasklist file: {}. Exiting.".format(
                self.tasklist))

    def __nest_postprocess(self):
        """Nest postprocessing."""
        import nest.timeGraphPlotter as TGP
        if self.config.timegraphs:
            tgp = TGP.timeGraphPlotter(self.config)
            tgp.plot_all()

        # if self.config.histograms

    def __auryn_postprocess(self):
        """Auryn postprocessing."""
        import auryn

    def main(self):
        """Do everything."""
        self.__load_config()
        if self.simulator == "nest":
            self.__nest_postprocess()
        elif self.simulator == "auryn":
            self.__auryn_postprocess()
        else:
            sys.exit("Postprocessing for {} not yet implemented.".format(
                self.simulator))


def usage():
    """Print usage."""
    print("Wrong arguments.", file=sys.stderr)
    print("Usage:", file=sys.stderr)
    sys.exit("\t{} auryn|nest".format(sys.argv[0]))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        runner = Postprocess(sys.argv[1])
        runner.main()
    else:
        usage()
