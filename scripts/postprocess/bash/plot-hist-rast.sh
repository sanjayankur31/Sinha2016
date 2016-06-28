#!/bin/bash

# Copyright 2015 Ankur Sinha 
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# File : 
#

NE=8000
NI=2000
NN=7200
NP=800
NR=400
ND=200
NSTIM=1000
RECALLTIME=4000000
SRC_DIR="/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016"
RASTERS_DIR="rasters"
HISTOGRAMS_DIR="histograms"

mkdir "$HISTOGRAMS_DIR"
mv spikes-E.gdf $HISTOGRAMS_DIR
mv spikes-I.gdf $HISTOGRAMS_DIR

    pushd "$HISTOGRAMS_DIR"
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 1. 30. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 85. 100. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 850. 870. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 2000. 2005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 4000. 4005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 5000. 5005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 5990. 5995. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 1. 30. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 85. 100. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 850. 870. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 2000. 2005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 4000. 4005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 5000. 5005. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 5990. 5995. )

        wait

        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  1. 30.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  85. 100.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  850. 870.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  2000. 2005.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  4000. 4005.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  5000. 5005.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  5990. 5995.

        echo "Running gnuplot on histograms."
        for i in *.plt; do gnuplot "$i"; done

    popd

echo "Plotting rasters"
mkdir "$RASTERS_DIR"
    pushd $RASTERS_DIR
        mv ../$HISTOGRAMS_DIR/spikes-E.gdf .
        mv ../$HISTOGRAMS_DIR/spikes-I.gdf .

        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 85.0 100.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 5990.0 5995.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 5000.0 5005.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 4000.0 4005.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 2000.0 2005.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 850.0 870.0
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 1.0 30

        echo "Running gnuplot on rasters."
        for i in *.plt; do gnuplot "$i"; done
    popd

