# load number of ranks
load 'settings.plt'

set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time in seconds"
set ylabel "Mean Synaptic weight (nS)"

set output "EE-weight.png"
set title "Mean Synaptic weight EE"
plot for [i=0:ranks-1] "00-synaptic-weights-".i.".txt" using 1 with lines lw 6 title ""

set output "EI-weight.png"
set title "Mean Synaptic weight EI"
plot for [i=0:ranks-1] "00-synaptic-weights-".i.".txt" using 2 with lines lw 6 title ""

set output "II-weight.png"
set title "Mean Synaptic weight II"
plot for [i=0:ranks-1] "00-synaptic-weights-".i.".txt" using ($3*-1) with lines lw 6 title ""

set output "IE-weight.png"
set title "Mean Synaptic weight IE"
plot for [i=0:ranks-1] "00-synaptic-weights-".i.".txt" using ($4*-1) with lines lw 6 title ""

set yrange [0:50]
set output "all-weights.png"
set title "Mean Synaptic weight EE"
plot for [i=0:ranks-1] "00-synaptic-weights-".i.".txt" using 1 with lines lw 6 title "", "00-synaptic-weights-".i.".txt" using 2 with lines lw 6 title "", "00-synaptic-weights-".i.".txt" using ($3*-1) with lines lw 6 title "", "00-synaptic-weights-".i.".txt" using ($4*-1) with lines lw 6 title ""
