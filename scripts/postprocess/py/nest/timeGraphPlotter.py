#!/usr/bin/env python3
"""
Plot the main time graphs.

File: plot-time-graphs.py

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

import os
import pandas
import subprocess
import sys


class timeGraphPlotter:

    """Plot the main time graphs."""

    def __init__(self, config):
        """Initialise."""
        self.config = config
        try:
            __import__('Gnuplot')
        except ImportError:
            print("Could not import Gnuplot module.", file=sys.stderr)
        else:
            self.plotter = Gnuplot.Gnuplot()
            self.plotter.reset()

            if (
                os.path.isfile(self.config.filenameRatesE) and
                os.stat(self.config.filenameRatesE).st_size != 0
            ):
                ratesE = pandas.load_csv(self.config.filenameRatesE, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineE = Gnuplot.data(ratesE.values[:0], ratesE.values[:1],
                                          title="E", with_="lines lw 4")
            else:
                self.lineE = [0, 0]

            if (
                os.path.isfile(self.config.filenameRatesI) and
                os.stat(self.config.filenameRatesI).st_size != 0
            ):
                ratesI = pandas.load_csv(self.config.filenameRatesI, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineI = Gnuplot.data(ratesI.values[:0], ratesI.values[:1],
                                          title="I", with_="lines lw 4")
            else:
                self.lineI = [0, 0]

            if (
                os.path.isfile(self.config.filenameRatesR) and
                os.stat(self.config.filenameRatesR).st_size != 0
            ):
                ratesR = pandas.load_csv(self.config.filenameRatesR, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineR = Gnuplot.data(ratesR.values[:0], ratesR.values[:1],
                                          title="R", with_="lines lw 4")
            else:
                self.lineR = [0, 0]

            if (
                os.path.isfile(self.config.filenameBatesB) and
                os.stat(self.config.filenameBatesB).st_size != 0
            ):
                ratesB = pandas.load_csv(self.config.filenameBatesB, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineB = Gnuplot.data(ratesB.values[:0], ratesB.values[:1],
                                          title="B", with_="lines lw 4")
            else:
                self.lineB = [0, 0]

            if (
                os.path.isfile(self.config.filenameSatesS) and
                os.stat(self.config.filenameSatesS).st_size != 0
            ):
                ratesS = pandas.load_csv(self.config.filenameSatesS, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineS = Gnuplot.data(ratesS.values[:0], ratesS.values[:1],
                                          title="S", with_="lines lw 4")
            else:
                self.lineS = [0, 0]

            if (
                os.path.isfile(self.config.filenameLatesL) and
                os.stat(self.config.filenameLatesL).st_size != 0
            ):
                ratesL = pandas.load_csv(self.config.filenameLatesL, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineL = Gnuplot.data(ratesL.values[:0], ratesL.values[:1],
                                          title="L", with_="lines lw 4")
            else:
                self.lineL = [0, 0]

            if (
                os.path.isfile(self.config.filenamePatesP) and
                os.stat(self.config.filenamePatesP).st_size != 0
            ):
                ratesP = pandas.load_csv(self.config.filenamePatesP, sep='\s+',
                                         dtype=float, lineterminator="\n",
                                         skipinitialspace=True, header=None,
                                         index_col=None, names=None)
                self.lineP = Gnuplot.data(ratesP.values[:0], ratesP.values[:1],
                                          title="P", with_="lines lw 4")
            else:
                self.lineP = [0, 0]

    def __plot_main(self):
        """Main plot with everything in it."""
        self.plotter('set term pngcairo font "OpenSans, 28" size 1920,1028')
        self.plotter.title("Mean firing rate for all available neuron sets")
        self.plotter.xlabel("Time (ms)")
        self.plotter.ylabel("Firing rate (Hz)")
        self.plotter("set yrange [0:200]")
        self.plotter("set ytics border nomirror 20")
        self.plotter("set xtics border nomirror")
        self.plotter.plot(self.lineE, self.lineI, self.lineB, self.lineL,
                          self.lineP, self.lineR, self.lineS)
        self.hardcopy(filename="firing-rate-all.png")

    def __plot_individuals(self):
        """Main plot with everything in it."""
        # TODO after Gnuplot is py3 compatible

    def __plot_I_E(self):
        """Plot one for I and E."""
        self.plotter('set term pngcairo font "OpenSans, 28" size 1920,1028')
        self.plotter.title("Mean firing rate for all I E neuron sets")
        self.plotter.xlabel("Time (ms)")
        self.plotter.ylabel("Firing rate (Hz)")
        self.plotter("set yrange [0:200]")
        self.plotter("set ytics border nomirror 20")
        self.plotter("set xtics border nomirror")
        self.plotter.plot(self.lineE, self.lineI)
        self.hardcopy(filename="firing-rate-I-E.png")

    def __plot_P_B(self):
        """Plot one for I and E."""
        self.plotter('set term pngcairo font "OpenSans, 28" size 1920,1028')
        self.plotter.title("Mean firing rate for all P B neuron sets")
        self.plotter.xlabel("Time (ms)")
        self.plotter.ylabel("Firing rate (Hz)")
        self.plotter("set yrange [0:200]")
        self.plotter("set ytics border nomirror 20")
        self.plotter("set xtics border nomirror")
        self.plotter.plot(self.lineP, self.lineB)
        self.hardcopy(filename="firing-rate-P-B.png")

    def plot_all(self):
        """Plot them all."""
        try:
            __import__('Gnuplot')
        except ImportError:
            print("Could not import Gnuplot. Using binary and plotting file.",
                  file=sys.stderr)
        else:
            self.__plot_main()
            self.__plot_individuals()
            self.__plot_I_E()
            self.__plot_P_B()

    def plot_using_gnuplot_binary(self):
        """Use the binary because it doesnt support py3."""
        args = ('/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/' +
                'scripts/postprocess/py/nest/plot-firing-rates.plt')
        subprocess.call(['gnuplot',
                         args])

if __name__ == "__main__":
    runner = timeGraphPlotter()
