OUT=perf.csv
#rm ${OUT}

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

mb=28
#Taking values from terminal

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

oh=$(((ifh + 2*pad_h - kh)/stride + 1))
ow=$(((ifw + 2*pad_w - kw)/stride + 1))


config="g1mb${mb}ic${nIfm}ih${ifh}iw${ifw}oc${nOfm}oh${56}ow${56}kh${kh}kw${kw}sh${stride}sw${stride}ph${pad_h}pw${pad_w}n"

#echo -n $config, >> ${OUT}
GFLOPS=`mkl-dnn/build/tests/benchdnn/benchdnn --conv --mode=p --dir=FWD_D --cfg=f32 $config"resnet_50:conv1" | grep "nresnet_50:conv1"| cut -d"," -f10`
echo -n $GFLOPS >> ${OUT}
echo "" >> ${OUT}

echo >> ${OUT}
