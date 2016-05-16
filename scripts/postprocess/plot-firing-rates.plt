set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons (Hz)"
set yrange [0:200]
set ytics border nomirror 20
set xtics border nomirror

set output "firing-rate-recall.png"
set title "Firing rate for recall neurons"
plot "firing-rate-recall.gdf" with lines lw 4 title "R";

set output "firing-rate-pattern.png"
set title "Firing rate for pattern neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P";

set output "firing-rate-background.png"
set title "Firing rate for background neurons"
plot "firing-rate-background.gdf" with lines lw 4 title "N";

set output "firing-rate-ExtE.png"
set title "Firing rate for Ext neurons"
plot "firing-rate-ExtE.gdf" with lines lw 4 title "ExtE";

set output "firing-rate-pattern-background.png"
set title "Firing rate for pattern and background neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-background.gdf" with lines lw 4 title "N";

set output "firing-rate-E.png"
set title "Firing rate for E neurons"
plot "firing-rate-E.gdf" with lines lw 4 title "E";

set output "firing-rate-I.png"
set title "Firing rate for I neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I";

set output "firing-rate-I-E.png"
set title "Firing rate for neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E";

set output "firing-rate-all.png"
set title "Firing rate for various neuron sets"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E" , "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-background.gdf" with lines lw 4 title "N", "firing-rate-recall.gdf" with lines lw 4 title "R", "firing-rate-lesioned.gdf" with lines lw 4 title "L";

set output "firing-rate-pattern-recall.png"
set title "Firing rate for pattern, recall, and lesioned neurons"
plot "firing-rate-pattern.gdf" with lines lw 4 title "P", "firing-rate-recall.gdf" with lines lw 4 title "R", "firing-rate-lesioned.gdf" with lines lw 4 title "L";

set output "firing-rate-lesioned.png"
set title "Firing rate for lesioned neurons"
plot "firing-rate-lesioned.gdf" with lines lw 4 title "L";
