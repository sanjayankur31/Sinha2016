#!/usr/bin/env python
"""
Unittests.

File: test_structural_p.py

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
import re


class TestStructuralPlasticity(unittest.TestCase):

    """Tests for Sinha 2016."""

    @classmethod
    def setUpClass(cls):
        """Setup."""
        cls.stabtime = 40.
        cls.sp_update_interval = 20.
        cls.sim = Sinha2016()
        cls.sim.populations = {'E': 80, 'I': 20, 'P': 8, 'R': 4,
                               'STIM': 10, 'Poisson': 1}
        cls.sim.setup_plasticity(True, True)
        cls.sim.prerun_setup(step=False, stabilisation_time=cls.stabtime,
                             sp_update_interval=cls.sp_update_interval,
                             recording_interval=10.)
        nest.set_verbosity('M_FATAL')
        cls.prefixes = ['spikes-E', 'spikes-I', 'spikes-pattern',
                        'spikes-background', 'spikes-recall',
                        'spikes-deaffed-pattern', 'spikes-deaffed-bg-E',
                        'spikes-deaffed-bg-I', 'spikes-stim',
                        '00-synaptic-weights-EE', '00-synaptic-weights-EI',
                        '00-synaptic-weights-II', '00-synaptic-weights-IE',
                        '01-calcium-E', '01-calcium-I',
                        'patternneurons',
                        'deaffed-patternneurons',
                        'non-deaffed-patternneurons',
                        'backgroundneurons',
                        'deaffed-backgroundneurons',
                        'non-deaffed-backgroundneurons',
                        'deaffed-Ineurons',
                        'non-deaffed-Ineurons',
                        'recallneurons',
                        '02-synaptic-elements-totals-E',
                        '02-synaptic-elements-totals-I',
                        '03-synaptic-elements-E',
                        '03-synaptic-elements-I',
                        ]

    @classmethod
    def tearDownClass(cls):
        """Tear down."""
        filelist = os.listdir('.')
        for f in filelist:
            if ".gdf" in f:
                os.unlink(f)
            if ".txt" in f:
                os.unlink(f)

    def test_01_neuron_creation(self):
        """Test neurons are created properly."""
        self.assertEqual(len(self.__class__.sim.neuronsE),
                         self.__class__.sim.populations['E'])
        self.assertEqual(len(self.__class__.sim.neuronsI),
                         self.__class__.sim.populations['I'])
        self.assertEqual(len(self.__class__.sim.poissonExtE),
                         self.__class__.sim.populations['Poisson'])
        self.assertEqual(len(self.__class__.sim.poissonExtI),
                         self.__class__.sim.populations['Poisson'])

    def test_02_simrun(self):
        """Test a complete sim run."""
        self.__class__.sim.stabilise()
        self.__class__.sim.store_pattern()
        self.__class__.sim.stabilise()
        self.__class__.sim.deaff_last_pattern()
        self.__class__.sim.enable_rewiring()
        self.__class__.sim.stabilise()
        self.__class__.sim.recall_last_pattern(10.)
        self.__class__.sim.close_files()
        self.assertEqual(nest.GetKernelStatus()['time'],
                         1000. * (3 * self.__class__.stabtime + 10.))

    def test_03_outputfiles(self):
        """Test output files."""
        filelist = os.listdir('.')
        checklist = []
        for f in filelist:
            for entry in self.__class__.prefixes:
                if re.match(entry, f):
                    checklist.append(entry)
                    break

        self.assertEqual(set(checklist), set(self.__class__.prefixes))


if __name__ == '__main__':
    unitttest.main()
