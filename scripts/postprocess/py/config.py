#!/usr/bin/env python3
"""
Config postprocess variables.

File: config.py

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

import configparser


class Config:

    """Config post process globals."""

    def __init__(self, taskfile='config.ini'):
        """Initialise."""
        self.timegraphs = True
        self.snr = False
        self.histogram_timelist = [0.]
        self.snr_time = 0.
        self.taskfile = taskfile

        self.filenameE = ""
        self.filenameI = ""
        self.filenameR = ""
        self.filenameB = ""
        self.filenameS = ""
        self.filenameL = ""
        self.filenameP = ""
        self.filenameRatesE = ""
        self.filenameRatesI = ""
        self.filenameRatesR = ""
        self.filenameRatesB = ""
        self.filenameRatesS = ""
        self.filenameRatesL = ""
        self.filenameRatesP = ""

        parser = configparser.ConfigParser()
        parser.read(self.taskfile)

        # all the different neuron sets
        self.neuronsE = parser['default']['neuronsE']
        self.neuronsI = parser['default']['neuronsI']
        self.neuronsR = parser['default']['neuronsR']
        self.neuronsB = parser['default']['neuronsB']
        self.neuronsS = parser['default']['neuronsS']
        self.neuronsL = parser['default']['neuronsL']
        self.neuronsP = parser['default']['neuronsP']
        self.filenameE = parser['default']['filenameE']
        self.filenameI = parser['default']['filenameI']
        self.filenameR = parser['default']['filenameR']
        self.filenameB = parser['default']['filenameB']
        self.filenameS = parser['default']['filenameS']
        self.filenameL = parser['default']['filenameL']
        self.filenameP = parser['default']['filenameP']
        self.filenameRatesE = parser['default']['filenameRatesE']
        self.filenameRatesI = parser['default']['filenameRatesI']
        self.filenameRatesR = parser['default']['filenameRatesR']
        self.filenameRatesB = parser['default']['filenameRatesB']
        self.filenameRatesS = parser['default']['filenameRatesS']
        self.filenameRatesL = parser['default']['filenameRatesL']
        self.filenameRatesP = parser['default']['filenameRatesP']

        self.timegraphs = parser['default'].getboolean('timegraphs')
        self.rows_per_read = int(parser['default']['rows_per_read'])

        # histograms and rasters
        self.histograms = parser['default'].getboolean('histograms')
        self.rasters = parser['default'].getboolean('rasters')
        self.histogram_timelist = [float(s) for s in
                                   parser['histograms']['times'].split()]
        self.store_rate_files = parser['histograms'].getboolean(
            'store_rate_files')
        self.store_raster_files = parser['histograms'].getboolean(
            'store_raster_files')

        # snr
        self.snr = parser['default'].getboolean('snr')
        self.snr_time = float(parser['snr']['times'])

if __name__ == "__main__":
    config = Config()
