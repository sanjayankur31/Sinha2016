set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time in seconds"
set ylabel "Mean Synaptic weight (nS)"

set title "IE weight time graph from an Auryn simulation"
set output "synaptic-weight-IE.png"

plot "ie.weightinfo" using 1:($2*10) with lines lw 6 title ""
