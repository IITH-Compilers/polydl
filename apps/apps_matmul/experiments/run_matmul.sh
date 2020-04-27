
export KMP_AFFINITY=granularity=fine,compact,1,28
export LD_LIBRARY_PATH=/nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install/lib:/nfs_home/stavarag/work/software/barvinok/isl_install/lib:$LD_LIBRARY_PATH

OUT=poly_perf.csv

check_correctness=1
PERF_DIR=perf_data
TEMP=temp
DATATYPESIZE=4

mkdir ${PERF_DIR}
mkdir ${TEMP}

iters=$1
M1=$2
N1=$3
K1=$4
M2_Tile=$5
N2_Tile=$6
K2_Tile=$7
M1_Tile=$8
N1_Tile=$9
K1_Tile=${10}
file=${11}

echo iters=$iters M1=$M1 N1=$N1 K1=$K1 M2_Tile=$M2_Tile N2_Tile=$N2_Tile K2_Tile=$K2_Tile M1_Tile=$M1_Tile N1_Tile=$N1_Tile K1_Tile=$K1_Tile file=$file

config=${iters}_${M1}_${N1}_${K1}_${M2_Tile}_${M2_Tile}_${N2_Tile}_${K2_Tile}_${M1_Tile}_${N1_Tile}_${K1_Tile}
echo config: $config

CONFIG_OUT=${PERF_DIR}/${file}_${OUT}
META_CONFIG_OUT=${PERF_DIR}/meta_${file}_${OUT}
#rm ${CONFIG_OUT}
#rm ${META_CONFIG_OUT}

export OMP_NUM_THREADS=28

(cd .. && make clean && make version_file=versions/$file MACROFLAGS="-DM1=$M1 -DN1=$N1 -DK1=$K1 -DNUM_ITERS=$iters -DM2_Tile=${M2_Tile} -DN2_Tile=${N2_Tile} -DK2_Tile=${K2_Tile} -DM1_Tile=${M1_Tile} -DN1_Tile=${N1_Tile} -DK1_Tile=${K1_Tile}")

../matmul &> run_output
GFLOPS=`cat run_output |  grep Real_GFLOPS |  cut -d= -f2`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
echo GFLOPS: $GFLOPS
echo ERROR: $ERROR

echo  "${config},${GFLOPS}"  >> ${CONFIG_OUT}
echo  "${config},${ERROR}" >> ${META_CONFIG_OUT}
