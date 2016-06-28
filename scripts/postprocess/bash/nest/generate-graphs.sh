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
# File : generate-graphs.sh
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

echo "Generating graphs"
pushd consolidated_files

source activate python3

echo "Processing E spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-E.gdf firing-rate-E.gdf $NE
touch firing-rate-E.gdf

echo "Processing I spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-I.gdf firing-rate-I.gdf $NI
touch firing-rate-I.gdf

echo "Processing pattern spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-pattern.gdf firing-rate-pattern.gdf $NP
touch firing-rate-pattern.gdf

echo "Processing background spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-background.gdf firing-rate-background.gdf $NN
touch firing-rate-background.gdf

echo "Processing recall spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-recall.gdf firing-rate-recall.gdf $NR
touch firing-rate-recall.gdf

echo "Processing deaffed spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-deaffed.gdf firing-rate-deaffed.gdf $ND
touch firing-rate-deaffed.gdf

echo "Processing Stim spikes"
python3 $SRC_DIR/scripts/postprocess/nest-spike2hz.py spikes-Stim.gdf firing-rate-Stim.gdf $NSTIM
touch firing-rate-Stim.gdf

echo "Plotting firing rate graphs"
gnuplot $SRC_DIR/scripts/postprocess/plot-firing-rates.plt

echo "Plotting EvsI graph"
gnuplot $SRC_DIR/scripts/postprocess/plot-EvsI.plt

mkdir "$HISTOGRAMS_DIR"
mv spikes-E.gdf $HISTOGRAMS_DIR
mv spikes-I.gdf $HISTOGRAMS_DIR

    pushd "$HISTOGRAMS_DIR"
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 1000. 30000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 85000. 100000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 850000. 870000. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 2000000. 2005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 4000000. 4005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 5000000. 5005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-E.gdf E $NE 5990000. 5995000. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 1000. 30000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 85000. 100000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 850000. 870000. )

        wait

        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 2000000. 2005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 4000000. 4005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 5000000. 5005000. )
        ( python3 $SRC_DIR/scripts/postprocess/nest-getFiringRates.py spikes-I.gdf I $NI 5990000. 5995000. )

        wait

        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  1000. 30000.
        python3 $SRC_DIR/scripts/postprocess/plot-histograms-time.py E I  85000. 100000.
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

popd

dirname=${PWD##*/}

echo "Renaming files."
rename "firing" "$dirname""-firing" consolidated_files/*.png
rename "hist" "$dirname""-hist" consolidated_files/$HISTOGRAMS_DIR/*.png
rename "rast" "$dirname""-rast" consolidated_files/$RASTERS_DIR/*.png
rename "EvsI" "$dirname""-EvsI" consolidated_files/*.png


echo "Moving files to test dir in repository."
mkdir -p $SRC_DIR/tests/$dirname/$RASTERS_DIR
mkdir -p $SRC_DIR/tests/$dirname/$HISTOGRAMS_DIR
cp consolidated_files/*.png $SRC_DIR/tests/$dirname/
cp consolidated_files/recall-snr.gdf $SRC_DIR/tests/$dirname/
cp consolidated_files/$HISTOGRAMS_DIR/*.png $SRC_DIR/tests/$dirname/$HISTOGRAMS_DIR
cp consolidated_files/$RASTERS_DIR/*.png $SRC_DIR/tests/$dirname/$RASTERS_DIR

source deactivate
