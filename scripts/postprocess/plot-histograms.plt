# plot histograms for various firing rates

# E neurons
#n=8000
#max=200.
#min=0.
#width=(max-min)/n
#hist(x,width)=width*floor(x/width)+width/2.0
#set term pngcairo font "OpenSans, 28" size 1920,1028
#set output "histogram-E.png"
#set xrange[min:max]
#set yrange[0:]
#set offset graph 0.05,0.05,0.05,0.0
#set xtics min,(max-min)/5,max
#set boxwidth width*0.9
#set style fill solid 0.5 #fillstyle
#set tics out nomirror
#set xlabel "Firing rate"
#set ylabel "Frequency"
#plot "firing-rate-E.gdf" u (hist($1,width)):(1.0) smooth freq w boxes title "E"
#
#reset
#n=2000
#max=200.
#min=0.
#width=(max-min)/n
#hist(x,width)=width*floor(x/width)+width/2.0
#set term pngcairo font "OpenSans, 28" size 1920,1028
#set output "histogram-I.png"
#set xrange[min:max]
#set yrange[0:]
#set offset graph 0.05,0.05,0.05,0.0
#set xtics min,(max-min)/5,max
#set boxwidth width*0.9
#set style fill solid 0.5 #fillstyle
#set tics out nomirror
#set xlabel "Firing rate"
#set ylabel "Frequency"
#plot "firing-rate-I.gdf" u (hist($1,width)):(1.0) smooth freq w boxes title "I"
#
#reset
#max=200.
#min=0.
#n1=8000
#width1=(max-min)/n1
#n2=2000
#width2=(max-min)/n2
#hist(x,width)=width*floor(x/width)+width/2.0
#set term pngcairo font "OpenSans, 28" size 1920,1028
#set output "histogram-E-I.png"
#set xrange[min:max]
#set yrange[0:]
#set offset graph 0.05,0.05,0.05,0.0
#set xtics min,(max-min)/5,max
#set boxwidth width*0.9
#set style fill solid 0.5 #fillstyle
#set tics out nomirror
#set xlabel "Firing rate"
#set ylabel "Frequency"
#plot "firing-rate-E.gdf" u (hist($1,width1)):(1.0) smooth freq w boxes title "E", "firing-rate-I.gdf" u (hist($1,width2)):(1.0) smooth freq w boxes title "I"; 


reset
n=200
max=200.
min=0.
width=(max-min)/n
hist(x,width)=width*floor(x/width)+width/2.0
set term pngcairo font "OpenSans, 28" size 1920,1028
set output "histogram-noise.png"
set xrange[min:max]
set yrange[0:]
set offset graph 0.05,0.05,0.05,0.0
set xtics min,20,max
set boxwidth width*0.9
set style fill solid 0.5 #fillstyle
set tics out nomirror
set xlabel "Firing rate"
set ylabel "Frequency"
plot "recall-firing-rate-noise.gdf" u (hist($1,width)):(1.0) smooth freq w boxes title "noise"

reset
n=200
max=200.
min=0.
width=(max-min)/n
hist(x,width)=width*floor(x/width)+width/2.0
set term pngcairo font "OpenSans, 28" size 1920,1028
set output "histogram-pattern.png"
set xrange[min:max]
set yrange[0:]
set offset graph 0.05,0.05,0.05,0.0
set xtics min,20,max
set boxwidth width*0.9
set style fill solid 0.5 #fillstyle
set tics out nomirror
set xlabel "Firing rate"
set ylabel "Frequency"
plot "recall-firing-rate-pattern.gdf" u (hist($1,width)):(1.0) smooth freq w boxes title "pattern"

reset
max=200.
min=0.
n1=200
width1=(max-min)/n1
n2=200
width2=(max-min)/n2
hist(x,width)=width*floor(x/width)+width/2.0
set term pngcairo font "OpenSans, 28" size 1920,1028
set output "histogram-pattern-noise.png"
set xrange[min:max]
set yrange[0:]
set offset graph 0.05,0.05,0.05,0.0
set xtics min,20,max
set boxwidth width*0.9
set style fill transparent solid 0.5 #fillstyle
set tics out nomirror
set xlabel "Firing rate"
set ylabel "Frequency"
plot "recall-firing-rate-pattern.gdf" u (hist($1,width1)):(1.0) smooth freq w boxes title "pattern", "recall-firing-rate-noise.gdf" u (hist($1,width2)):(1.0) smooth freq w boxes title "noise"; 

