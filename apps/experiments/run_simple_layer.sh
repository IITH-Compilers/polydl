#set -x
MB=28
export KMP_AFFINITY=granularity=fine,compact,1,28
export OMP_NUM_THREADS=$MB
LAYER_EXAMPLE_ICC=/nfs_home/stavarag/work/libxsmm/libxsmm_simple_icc_password-1234/libxsmm_simple_icc/layer_example_icc


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
TYPE=F
OUTPUT=layer_example_icc.output
DIR=simple_layer_perf
mkdir $DIR
COMP_OUT=compiler.csv
GEMM_OUT=gemm.csv

${LAYER_EXAMPLE_ICC} ${iters}   ${ifw}   ${ifh}  ${MB} ${nIfm}  ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${stride} ${TYPE} &> ${OUTPUT}

GFLOPS_comp=`cat $OUTPUT | grep GFLOPS_compiler | cut -d= -f2`
GFLOPS_gmm=`cat $OUTPUT | grep GFLOPS_gemm | cut -d= -f2`
echo ${config_num},${GFLOPS_comp} >> $DIR/${COMP_OUT}
echo ${config_num},${GFLOPS_gmm} >> $DIR/${GEMM_OUT}
