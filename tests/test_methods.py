#!/usr/bin/env python
"""
Test utility methods.

File: test_methods.py

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


class TestUtilityMethods(unittest.TestCase):

    """Test utility methods."""

    def test_01_get_synaptic_elements(self):
        """A sanity test to check if method gets syn elements correctly."""
        stabtime = 40.
        sp_update_interval = 20.
        sim = Sinha2016()
        sim.populations = {'E': 80, 'I': 20, 'P': 8, 'R': 4,
                           'STIM': 10, 'Poisson': 1}
        # give no input at all
        sim.poissonExtDict = {'rate': 0., 'origin': 0., 'start': 0.}
        sim.setup_plasticity(True, True)
        sim.prerun_setup(step=False, stabilisation_time=stabtime,
                         sp_update_interval=sp_update_interval,
                         recording_interval=10.)
        nest.set_verbosity('M_FATAL')
        sim.enable_rewiring()
