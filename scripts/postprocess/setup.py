#!/usr/bin/env python3
"""
Setup postprocess variables.

File: setup.py

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


class Setup:

    """Setup post process globals."""

    def __init__(self):
        """Initialise."""
        self.simulator = "nest"
        self.timegraphs = True
        self.snr = False
        self.histogram_timelist = [0.]
        self.snr_time = 0.
        self.taskfile = 'tasklist.ini'

        self.filenameE = ""
        self.filenameI = ""
        self.filenameR = ""
        self.filenameB = ""
        self.filenameN = ""
        self.filenameL = ""

        parser = configparser.ConfigParser()

        print("Reading task file: {}".format(self.taskfile))
        parser.read(self.taskfile)

        # all the filenames
        self.filenameE = parser['default']['filenameE']
        self.filenameI = parser['default']['filenameI']
        self.filenameR = parser['default']['filenameR']
        self.filenameB = parser['default']['filenameB']
        self.filenameN = parser['default']['filenameN']
        self.filenameL = parser['default']['filenameL']

        self.simulator = parser['default']['simulator']
        self.timegraphs = bool(parser['default']['timegraphs'])

        # histograms and rasters
        self.histogram_timelist = [float(s) for s in
                                   parser['histograms']['times'].split()]

        # snr
        self.snr = bool(parser['default']['snr'])
        self.snr_time = float(parser['snr']['times'])

if __name__ == "__main__":
    config = Setup()
