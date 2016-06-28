set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons (Hz)"
set yrange [0:200]
set ytics border nomirror 20
set xtics border nomirror

set output "firing-rate-E-auryn-nest.png"
set title "Firing rate for auryn and nest E neuron sets"
plot "firing-rate-E-auryn.gdf" with lines lw 4 title "E auryn" , "firing-rate-E-nest.gdf" with lines lw 4 title "E nest";

set output "firing-rate-I-auryn-nest.png"
set title "Firing rate for auryn and nest I neuron sets"
plot "firing-rate-I-auryn.gdf" with lines lw 4 title "I auryn" , "firing-rate-I-nest.gdf" with lines lw 4 title "I nest";

set output "firing-rate-auryn-nest.png"
set title "Firing rate for auryn and nest neuron sets"
plot "firing-rate-I-auryn.gdf" with lines lw 4 title "I auryn", "firing-rate-E-auryn.gdf" with lines lw 4 title "E auryn" , "firing-rate-I-nest.gdf" with lines lw 4 title "I nest", "firing-rate-E-nest.gdf" with lines lw 4 title "E nest";
