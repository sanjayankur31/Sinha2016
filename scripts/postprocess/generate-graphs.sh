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

echo "Plotting rasters"
mkdir "$RASTERS_DIR"
    pushd "$RASTERS_DIR"
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 0. 50.
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 80. 100.
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 850. 900.
        python3 $SRC_DIR/scripts/postprocess/plot-rasters.py 5000. 5050.

        cp ../spikes*gdf .
        for i in *.plt; do gnuplot "$i"; done
        rm spikes*gdf
    popd

mkdir "$HISTOGRAMS_DIR"
    pushd "$HISTOGRAMS_DIR"
        cp ../spikes*gdf .

        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 0.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 50.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 80.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 100.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 850.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-E.gdf spikes-I.gdf $NE $NI 900.
        python3 $SRC_DIR/scripts/postprocess/calculateSnapshotStats.py spikes-pattern.gdf spikes-background.gdf $NP $NN $RECALLTIME
        for i in *.plt; do gnuplot "$i"; done
        rm spikes*gdf
    popd
popd

dirname=${PWD##*/}

echo "Renaming files."
rename "firing" "$dirname""-firing" consolidated_files/*.png
rename "hist" "$dirname""-hist" consolidated_files/*.png
rename "EvsI" "$dirname""-EvsI" consolidated_files/*.png


echo "Moving files to test dir in repository."
mkdir -p $SRC_DIR/tests/$dirname/$RASTERS_DIR
cp consolidated_files/*.png $SRC_DIR/tests/$dirname/
cp consolidated_files/recall-snr.gdf $SRC_DIR/tests/$dirname/
cp consolidated_files/$RASTERS_DIR/*.png $SRC_DIR/tests/$dirname/$RASTERS_DIR

source deactivate
