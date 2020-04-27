OUT=perf.csv
rm ${OUT}

#Default values.
iters=1
ifw=56
ifh=56
nIfm=64
nOfm=256
kw=1
kh=1
pad_w=0
pad_h=0
stride=1


iters=$1
ifw=$2
ifh=$3
nIfm=$4
nOfm=$5
kw=$6
kh=$7
pad_w=$8
pad_h=$9
stride=${10}
config_num=${11}

config="${iters} ${ifw} ${ifh} ${nIfm} ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${stride}"


GFLOPS=`python3 simplified_tune_relay_x86_custom_generic.py $config | grep "GFLOPS" |rev| cut -d "|" -f3|rev | cut -d " " -f11`

echo -n ${config_num},$GFLOPS, >> ${OUT}
echo "" >> ${OUT}

