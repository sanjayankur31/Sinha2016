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
        self.configfile = "config-{}.ini".format(simulator)

    def __load_config(self):
        """Load configuration file."""
        if os.path.isfile(self.configfile):
            self.config = Config(self.configfile)
            print("Config file {} loaded successfully.".format(
                self.configfile))
        else:
            sys.exit("Could not find config file: {}. Exiting.".format(
                self.configfile))

    def __nest_postprocess(self):
        """Nest postprocessing."""
        if self.config.timegraphs:
            import nest.timeGraphPlotter as TGP
            tgp = TGP.timeGraphPlotter(self.config)
            tgp.plot_all()

        if self.config.histograms:
            import nest.dualHistogramPlotter as pltH
            import nest.getFiringRates as rg
            rateGetterE = rg.getFiringRates()
            if rateGetterE.setup(self.config.filenameE, 'E',
                                 self.config.neuronsE,
                                 self.config.rows_per_read):
                rateGetterE.run(self.config.histogram_timelist)

            rateGetterI = rg.getFiringRates()
            if rateGetterI.setup(self.config.filenameI, 'I',
                                 self.config.neuronsI,
                                 self.config.rows_per_read):
                rateGetterI.run(self.config.histogram_timelist)

            plotterEI = pltH.dualHistogramPlotter()
            if plotterEI.setup('E', 'I', self.config.neuronsE,
                               self.config.neuronsI):
                plotterEI.run()

            rateGetterB = rg.getFiringRates()
            if rateGetterB.setup(self.config.filenameB, 'B',
                                 self.config.neuronsB,
                                 self.config.rows_per_read):
                rateGetterB.run(self.config.histogram_timelist)

            rateGetterS = rg.getFiringRates()
            if rateGetterS.setup(self.config.filenameS, 'S',
                                 self.config.neuronsS,
                                 self.config.rows_per_read):
                rateGetterS.run(self.config.histogram_timelist)

            plotterBS = pltH.dualHistogramPlotter()
            if plotterBS.setup('B', 'S', self.config.neuronsB,
                               self.config.neuronsS):
                plotterBS.run()

        if self.config.rasters:
            import nest.dualRasterPlotter as pltR
            rasterPlotterEI = pltR.dualRasterPlotter()
            if rasterPlotterEI.setup('E', 'I', self.config.neuronsE,
                                     self.config.neuronsI,
                                     self.config.rows_per_read):
                rasterPlotterEI.run(self.config.histogram_timelist)


    def main(self):
        """Do everything."""
        self.__load_config()
        self.__nest_postprocess()

def usage():
    """Print usage."""
    print("Wrong arguments.", file=sys.stderr)
    print("Usage:", file=sys.stderr)
    sys.exit("\t{} nest".format(sys.argv[0]))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        runner = Postprocess(sys.argv[1])
        runner.main()
    else:
        usage()
