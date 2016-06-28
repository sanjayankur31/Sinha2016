# load number of ranks
load 'settings.plt'

set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean calcium concentrations"
set ytics border nomirror
set xtics border nomirror

set output "calcium-concentration-E.png"
set title "Calcium concentration for E neuron sets"
plot for [i=0:ranks-1] "calcium-".i.".txt" using 1 with lines lw 6 title ""

set output "calcium-concentration-I.png"
set title "Calcium concentration for I neuron sets"
plot for [i=0:ranks-1] "calcium-".i.".txt" using 2 with lines lw 6 title ""
