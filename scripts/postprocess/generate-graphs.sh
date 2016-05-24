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

echo "Processing SNR information"
python3 $SRC_DIR/scripts/postprocess/calculateSNR.py spikes-pattern.gdf spikes-background.gdf $NP $NN $RECALLTIME
echo "Plotting histograms"
gnuplot $SRC_DIR/scripts/postprocess/plot-histograms.plt

popd

dirname=${PWD##*/}

mkdir $SRC_DIR/tests/$dirname
cp consolidated_files/*.png $SRC_DIR/tests/$dirname/
cp consolidated_files/recall-snr.gdf $SRC_DIR/tests/$dirname/

source deactivate
