set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons (Hz)"
set yrange [0:200]
set ytics border nomirror 20
set xtics border nomirror

set output "firing-rate-recall.png"
set title "Firing rate for recall neurons"
plot "firing-rate-recall.gdf" with lines lw 4 title "R", 8 with lines lw 2 title "T";

set output "firing-rate-pattern.png"
set title "Firing rate for pattern neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P", 8 with lines lw 2 title "T";

set output "firing-rate-background.png"
set title "Firing rate for background neurons"
plot "firing-rate-background.gdf" with lines lw 4 title "N", 8 with lines lw 2 title "T";

set output "firing-rate-Stim.png"
set title "Firing rate for Stim neurons"
plot "firing-rate-Stim.gdf" with lines lw 4 title "Stim", 8 with lines lw 2 title "T";

set output "firing-rate-pattern-background.png"
set title "Firing rate for pattern and background neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-background.gdf" with lines lw 4 title "N", 8 with lines lw 2 title "T";

set output "firing-rate-E.png"
set title "Firing rate for E neurons"
plot "firing-rate-E.gdf" with lines lw 4 title "E", 8 with lines lw 2 title "T";

set output "firing-rate-I.png"
set title "Firing rate for I neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", 8 with lines lw 2 title "T";

set output "firing-rate-I-E.png"
set title "Firing rate for neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E", 8 with lines lw 2 title "T";

# zoomed in graphs
unset yrange
set xrange [0:250]
set output "firing-rate-I-E-zoomed-stage1.png"
set title "Firing rate for neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E", 8 with lines lw 2 title "T";

set xrange [750:1000]
set output "firing-rate-I-E-zoomed-stage2.png"
set title "Firing rate for neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E", 8 with lines lw 2 title "T";

unset xrange
set yrange [0:200]
set output "firing-rate-all.png"
set title "Firing rate for various neuron sets"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E" , "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-background.gdf" with lines lw 4 title "N", "firing-rate-recall.gdf" with lines lw 4 title "R", "firing-rate-deaffed.gdf" with lines lw 4 title "D", 8 with lines lw 2 title "T";

set output "firing-rate-pattern-recall.png"
set title "Firing rate for pattern, recall, and deaffed neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-recall.gdf" with lines lw 4 title "R", "firing-rate-deaffed.gdf" with lines lw 4 title "D", 8 with lines lw 2 title "T";

set output "firing-rate-deaffed.png"
set title "Firing rate for deaffed neurons"
plot "firing-rate-deaffed.gdf" with lines lw 4 title "D", 8 with lines lw 2 title "T";
