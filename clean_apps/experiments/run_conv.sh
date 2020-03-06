#!/bin/bash
set -x
compute_padding()
{

MAX_BLOCK=64
input=$1
output=$input
MIN_BLOCK=16

#Padding
for BLOCK in 16 32 64
do
        if [ `expr $input % $BLOCK` -eq 0 ]
        then
          break
        fi

        if [ $input -lt $BLOCK ]
        then
          let output=$BLOCK
          break
        else
          if [ $BLOCK -eq $MAX_BLOCK -a $input -gt $MAX_BLOCK ]
          then
                let output=( $input + $MIN_BLOCK -  $input % $MIN_BLOCK  )
          fi
        fi
done

echo $output
}

compute_GEMM_BLOCK()
{
  nIfm=$1
  nOfm=$2

  #echo In compute_GEMM_BLOCK. nIfm: $nIfm nOfm: $nOfm

  for BLOCK in 64 32 16
  do
        if [ `expr $nIfm % $BLOCK` -eq 0 -a `expr $nOfm % $BLOCK` -eq 0 ]
        then
          #echo $BLOCK divides them both
          echo $BLOCK
          break
        fi
  done
}

export KMP_AFFINITY=granularity=fine,compact,1,28
export LD_LIBRARY_PATH=/nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install/lib:/nfs_home/stavarag/work/software/barvinok/isl_install/lib:$LD_LIBRARY_PATH

OUT=poly_perf.csv

check_correctness=1
PERF_DIR=perf_data
CONFIG_DIR=configs
TEMP=temp
GEMM_VERSIONS='0 1 2 3 4 5'
NONGEMM_VERSIONS='20 21 22 23 24 25'    
DATATYPESIZE=4

mkdir ${PERF_DIR}
mkdir ${TEMP}
       iters=$1
       ifw=$2
       ifh=$3
       orig_nIfm=$4
       orig_nOfm=$5
       kw=$6
       kh=$7
       pad_w=$8
       pad_h=$9
       stride=${10}
       config_num=${11}
       files_v=${12}

       let ofh="($ifh + 2 * $pad_h - $kh)/$stride + 1"
       let ofw="($ifw + 2 * $pad_w - $kw)/$stride + 1"
       ofhp=${ofh}
       ofwp=${ofw}
       arg_images=1
       ifhp=${ifh}
       ifwp=${ifw}


nIfm=$(compute_padding $orig_nIfm)
echo orig_nIfm: $orig_nIfm nIfm: $nIfm

nOfm=$(compute_padding $orig_nOfm)
echo orig_nOfm: $orig_nOfm  nOfm: $nOfm

GEMM_BLOCK=$(compute_GEMM_BLOCK $nIfm $nOfm)
echo GEMM_BLOCK: $GEMM_BLOCK

	for images in 1 28
	do
	        CONFIG_OUT=${PERF_DIR}/${config_num}_${images}_${OUT}
		META_CONFIG_OUT=${PERF_DIR}/meta_${config_num}_${images}_${OUT}
	        rm ${CONFIG_OUT}
		rm ${META_CONFIG_OUT}

		if [ ${images} -eq 1 ]
		then
			CACHE_CONFIG='L1 32768 L2 1048576 L3 40370176'
		fi

                if [ ${images} -eq 28 ]
                then
                        CACHE_CONFIG='L1 32768 L2 1048576 L3 1441792'
                fi


		export OMP_NUM_THREADS=${images}
                #We will first do an actual run
                if echo "$files_v" | grep -q "tile";
                then
                        if [ `expr $nIfm % $GEMM_BLOCK` -eq 0 -a `expr $nOfm % $GEMM_BLOCK` -eq 0 ]
                        then

                        for (( T_oi=${ofw}; T_oi<= ${ofw}; T_oi=T_oi*4 ))
                        do
                        if [ `expr $ofw % $T_oi` -eq 0 ] 
                        then
                        for (( T_oj=${ofh}; T_oj >= 1; T_oj=T_oj/2 ))
                        do
                        if [ `expr $ofh % $T_oj` -eq 0 ] 
                        then
                        for (( T_ifm_tile=${nIfm}; T_ifm_tile<= ${nIfm}; T_ifm_tile=T_ifm_tile*2 ))
                        do
                        if [ `expr $nIfm % $T_ifm_tile` -eq 0 ]
                        then
                        for (( T_ofm_tile=${nOfm}; T_ofm_tile<= ${nOfm}; T_ofm_tile=T_ofm_tile*2 ))
                        do
                        if [ `expr $nOfm % $T_ofm_tile` -eq 0 ]
                        then

                                (cd .. && make clean && make version_file=${files_v} MACROFLAGS="-DSTRIDE_H=$stride -DSTRIDE_W=$stride -DT_oi=$T_oi -DT_oj=$T_oj -DT_ifm_tile=$T_ifm_tile -DT_ofm_tile=$T_ofm_tile -DGEMM_BLOCK=${GEMM_BLOCK}")
                                # do something
                                pwd >> log
                                echo  $T_oi " " $T_oj " " $T_ifm_tile " " $T_ofm_tile
                                ../conv2d ${iters} ${ifw} ${ifh} ${nIfm} ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${stride} ${images} 0 ${check_correctness} &> run_output
                                Orig_GFLOPS=`cat run_output |  grep Real_GFLOPS |  cut -d= -f2`
                                GFLOPS=$(echo "$Orig_GFLOPS * $orig_nIfm * $orig_nOfm  /  $nIfm / $nOfm"|bc )

                                Orig_NAIVE_GFLOPS=`cat run_output |  grep Naive_GFLOPS |  cut -d= -f2` 
                                NAIVE_GFLOPS=$(echo "$Orig_NAIVE_GFLOPS * $orig_nIfm * $orig_nOfm  /  $nIfm / $nOfm"|bc )
                                ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
                                rm ${TEMP}/temp.c

                                cp ./${files_v} ${TEMP}/temp.c


                                rm tile_sizes.c
                                echo "#define GEMM_BLOCK ${GEMM_BLOCK}" >> tile_sizes.c
                                echo "#define STRIDE_H ${stride}" >> tile_sizes.c
                                echo "#define STRIDE_W ${stride}" >> tile_sizes.c
                                echo "#define T_oi ${T_oi}" >> tile_sizes.c
                                echo "#define T_oj ${T_oj}" >> tile_sizes.c
                                echo "#define T_ifm_tile ${T_ifm_tile}" >> tile_sizes.c
                                echo "#define T_ofm_tile ${T_ofm_tile}" >> tile_sizes.c
                                cat temp/temp.c >> tile_sizes.c
                                mv tile_sizes.c ${TEMP}/temp.c
                                output_file=${TEMP}/temp.c_ws_stats.csv
                                rm ${output_file}
                                ../../data_reuse_analyzer/polyscientist --input ${TEMP}/temp.c --parameters "ofw ofh nIfm nOfm kw kh pad_w pad_h nImg ifwp ifhp ofwp ofhp : ${ofw} ${ofh} ${nIfm} ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${arg_images} ${ifwp} ${ifhp} ${ofwp} ${ofhp}"  --cachesizes "${CACHE_CONFIG}" --datatypesize $DATATYPESIZE --minout 
                                { echo -n "0_${T_oi}_${T_oj}_${T_ifm_tile}_${T_ofm_tile}_${GEMM_BLOCK},${GFLOPS}," ;  cat  ${output_file} ; } >> ${CONFIG_OUT}
                                echo  "${NAIVE_GFLOPS},${ERROR}" >> ${META_CONFIG_OUT}


                        fi
                        done
                        fi
                        done
                        fi
                        done
                        fi
                        done
                        fi
                else
                        if [ `expr $nIfm % $GEMM_BLOCK` -eq 0 -a `expr $nOfm % $GEMM_BLOCK` -eq 0 ]
                        then
                        (cd .. && make clean && make version_file=${files_v} MACROFLAGS="-DSTRIDE_H=$stride -DSTRIDE_W=$stride -DGEMM_BLOCK=${GEMM_BLOCK}") 	
                        ../conv2d ${iters} ${ifw} ${ifh} ${nIfm} ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${stride} ${images} 0 ${check_correctness} &> run_output
                        echo "Naive_GFLOPS =1094.3" >> run_output
                        Orig_GFLOPS=`cat run_output |  grep Real_GFLOPS |  cut -d= -f2`
                        GFLOPS=$(echo "$Orig_GFLOPS * $orig_nIfm * $orig_nOfm  /  $nIfm / $nOfm"|bc )
                        Orig_NAIVE_GFLOPS=`cat run_output |  grep Naive_GFLOPS |  cut -d= -f2`
                        NAIVE_GFLOPS=$(echo "$Orig_NAIVE_GFLOPS * $orig_nIfm * $orig_nOfm  /  $nIfm / $nOfm"|bc )
                        ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
                        rm ${TEMP}/temp.c
                        echo "#define GEMM_BLOCK ${GEMM_BLOCK}" >> ${TEMP}/temp.c
                        echo "#define STRIDE_H ${stride}" >> ${TEMP}/temp.c
                        echo "#define STRIDE_W ${stride}" >> ${TEMP}/temp.c


                        cat "../${files_v}" >> ${TEMP}/temp.c
                        echo `pwd`

                        output_file=${TEMP}/temp.c_ws_stats.csv
                        rm ${output_file}
                        ../../data_reuse_analyzer/polyscientist --input ${TEMP}/temp.c --parameters "ofw ofh nIfm nOfm kw kh pad_w pad_h nImg ifwp ifhp ofwp ofhp : ${ofw} ${ofh} ${nIfm} ${nOfm} ${kw} ${kh} ${pad_w} ${pad_h} ${arg_images}  ${ifwp} ${ifhp} ${ofwp} ${ofhp}"  --cachesizes "${CACHE_CONFIG}" --datatypesize $DATATYPESIZE --minout 

                        # exit 1
                        { echo -n "0_${GEMM_BLOCK},${GFLOPS}," ; cat ${output_file} ; } >> ${CONFIG_OUT}
                        echo  "${NAIVE_GFLOPS},${ERROR}" >> ${META_CONFIG_OUT}
                        fi
                fi
		../../scripts/polyrank ${CONFIG_OUT}  --noheader --perfseparaterow --usepessidata
	done

