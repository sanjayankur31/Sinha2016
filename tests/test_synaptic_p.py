#!/usr/bin/env python3
"""
Unittests.

File: test_model.py

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


import unittest
import sys
sys.path.append("src")
sys.argv.append('--quiet')
from Sinha2016 import Sinha2016
import os
import nest


class TestSynapticPlasticity(unittest.TestCase):

    """Tests for Sinha 2016."""

    def setUp(self):
        """Setup."""
        self.sim = Sinha2016()
        self.sim.populations = {'E': 80, 'I': 20, 'P': 8, 'R': 4,
                                'D': 2, 'STIM': 10, 'Poisson': 1}
        self.sim.setup_plasticity(False, True)
        self.sim.prerun_setup(step=False, stabilisation_time=200.,
                              recording_interval=10.)
        nest.set_verbosity('M_FATAL')

    def tearDown(self):
        """Tear down."""
        self.sim.close_files()
        filelist = os.listdir('.')
        for f in filelist:
            if "*.gdf" in f:
                os.unlink(f)
            if "*.txt" in f:
                os.unlink(f)

    def test_neuron_creation(self):
        """Test neurons are created properly."""
        self.assertEqual(len(self.sim.neuronsE), self.sim.populations['E'])
        self.assertEqual(len(self.sim.neuronsI), self.sim.populations['I'])
        self.assertEqual(len(self.sim.poissonExtE),
                         self.sim.populations['Poisson'])
        self.assertEqual(len(self.sim.poissonExtI),
                         self.sim.populations['Poisson'])

    def test_simrun(self):
        """Test a sim run."""
        self.sim.stabilise()
        self.assertEqual(nest.GetKernelStatus()['time'], 200000.)


if __name__ == '__main__':
    unitttest.main()
