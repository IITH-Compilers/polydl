export KMP_AFFINITY=granularity=fine,compact,1,28
export LD_LIBRARY_PATH=/nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install/lib:/nfs_home/stavarag/work/software/barvinok/isl_install/lib:$LD_LIBRARY_PATH

OUT=poly_perf.csv

test_config='100  7  7  1 1 1 1 0 0 1'
config1='1000  56  56  64  256 1 1 0 0 1'
config2='1000  56  56  64  64 1 1 0 0 1'
config3='1000  56  56  64  64 3 3 1 1 1'
config4='1000  56  56  256  64 1 1 0 0 1'
config5='1000  28  28  256   512 1 1 0 0 1'
config6='1000  28  28  256   128 1 1 0 0 1'
config7='1000  28  28  128   128 3 3 1 1 1'
config8='1000  28  28  128   512 1 1 0 0 1'
config9='1000  28  28  512   128 1 1 0 0 1'
config10='1000  14  14  512  1024 1 1 0 0 1'
config11='1000  14  14  512   256 1 1 0 0 1'
config12='1000  14  14  256   256 3 3 1 1 1'
config13='1000  14  14  256  1024 1 1 0 0 1'
config14='1000  14  14  1024   256 1 1 0 0 1'
config15='1000  7   7   1024  2048 1 1 0 0 1'
config16='1000  7   7   1024   512 1 1 0 0 1'
config17='1000  7   7   512   512 3 3 1 1 1'
config18='1000  7   7   512  2048 1 1 0 0 1'
config19='1000  7   7   2048   512 1 1 0 0 1'

GEMM_BLOCK=64
config_num=3
check_correctness=0
PERF_DIR=perf_data
CONFIG_DIR=configs
TEMP=temp
mkdir ${PERF_DIR}
mkdir ${TEMP}
for config in "$config3" "$config4" "$config5" "$config6" 
do
	for images in 1 28
	do
	        CONFIG_OUT=${PERF_DIR}/${config_num}_${images}_${OUT}
	        rm ${CONFIG_OUT}

		export OMP_NUM_THREADS=${images}
		for version in 2 3 4 5 1 0
		do
			params=( ${config} )
			ofw=${params[1]}
			ofh=${params[2]}
			nIfm=${params[3]}
			nOfm=${params[4]}
			#We will first do an actual run
			if [ $version -eq 0 -o $version -eq 1 ]
			then
				for (( T_oi=7; T_oi<= ${ofw}; T_oi=T_oi*4 ))
				do
				if [ `expr $ofw % $T_oi` -eq 0 ] 
				then
				for (( T_oj=7; T_oj<= ${ofh}; T_oj=T_oj*4 ))
				do
				if [ `expr $ofh % $T_oj` -eq 0 ] 
				then
				for (( T_ifm_tile=1; T_ifm_tile<= ${nIfm}; T_ifm_tile=T_ifm_tile*4 ))
                                do
				if [ `expr $nIfm % $T_ifm_tile` -eq 0 ]
				then
                                for (( T_ofm_tile=1; T_ofm_tile<= ${nOfm}; T_ofm_tile=T_ofm_tile*4 ))
                                do
                                if [ `expr $nOfm % $T_ofm_tile` -eq 0 ]
                                then

					(cd .. && make clean && make MACROFLAGS="-DT_oi=$T_oi -DT_oj=$T_oj -DT_ifm_tile=$T_ifm_tile -DT_ofm_tile=$T_ofm_tile")
     					# do something
					echo  $T_oi " " $T_oj " " $T_ifm_tile " " $T_ofm_tile
					GFLOPS=`../conv2d $config ${images} ${version} ${check_correctness} |  grep GFLOPS |  cut -d= -f2`

					rm ${TEMP}/temp.c
					if [ $version -eq 0 ] 
					then
					cp ../padded_conv_fp_stride_1_tiled_loop_order_0.c ${TEMP}/temp.c
					fi

					if [ $version -eq 1 ] 
					then
					cp ../padded_conv_fp_stride_1_tiled_loop_order_1.c ${TEMP}/temp.c
					fi

					rm tile_sizes.c
					echo "#define T_oi ${T_oi}" > tile_sizes.c
					echo "#define T_oj ${T_oj}" >> tile_sizes.c
					echo "#define T_ifm_tile ${T_ifm_tile}" >> tile_sizes.c
					echo "#define T_ofm_tile ${T_ofm_tile}" >> tile_sizes.c
					cat temp/temp.c >> tile_sizes.c
					mv tile_sizes.c ${TEMP}/temp.c
					config_file=${config_num}_${images}_conv_config.txt
					output_file=${TEMP}/temp.c${config_file}_ws_stats.csv
					rm ${output_file}
					../../data_reuse_analyzer/polyscientist --input ${TEMP}/temp.c --config ${CONFIG_DIR}/${config_file} --minout 
					echo -n "${version}_${T_oi}_${T_oj}_${T_ifm_tile}_${T_ofm_tile},${GFLOPS}," | cat - ${output_file} >> ${CONFIG_OUT}


				fi
				done
				fi
				done
				fi
				done
				fi
				done
				echo
			else
				(cd .. && make clean && make) 	
				GFLOPS=`../conv2d $config ${images} ${version} ${check_correctness} |  grep GFLOPS |  cut -d= -f2`
                                rm ${TEMP}/temp.c
                                if [ $version -eq 2 ]
                                then
                                cp ../padded_conv_fp_stride_1_libxsmm_core.c ${TEMP}/temp.c
                                fi

                                if [ $version -eq 3 ]
                                then
                                cp ../padded_conv_fp_stride_1_libxsmm_core2.c ${TEMP}/temp.c
                                fi

                                if [ $version -eq 4 ]
                                then
                                cp ../padded_conv_fp_stride_1_libxsmm_core3.c ${TEMP}/temp.c
                                fi

                                if [ $version -eq 5 ]
                                then
                                cp ../padded_conv_fp_stride_1_libxsmm_core4.c ${TEMP}/temp.c
                                fi

                                config_file=${config_num}_${images}_conv_config.txt
                                output_file=${TEMP}/temp.c${config_file}_ws_stats.csv
                                rm ${output_file}
                                ../../data_reuse_analyzer/polyscientist --input ${TEMP}/temp.c --config ${CONFIG_DIR}/${config_file} --minout
                                echo -n "${version},${GFLOPS}," | cat - ${output_file} >> ${CONFIG_OUT}
			fi
		done
		../../scripts/polyrank ${CONFIG_OUT}  --noheader --perfseparaterow
	done
	((config_num++))
done



