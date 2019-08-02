export KMP_AFFINITY=granularity=fine,compact,1,0
OUT=poly_perf.csv

config1='100  56  56  64  256 1 1 0 0 1'
config2='100  56  56  64  64 1 1 0 0 1'
config3='100  56  56  64  64 3 3 1 1 1'
config4='100  56  56  256  64 1 1 0 0 1'
config5='100  28  28  256   512 1 1 0 0 1'
config6='100  28  28  256   128 1 1 0 0 1'
config7='100  28  28  128   128 3 3 1 1 1'
config8='100  28  28  128   512 1 1 0 0 1'
config9='100  28  28  512   128 1 1 0 0 1'
config10='100  14  14  512  1024 1 1 0 0 1'
config11='100  14  14  512   256 1 1 0 0 1'
config12='100  14  14  256   256 3 3 1 1 1'
config13='100  14  14  256  1024 1 1 0 0 1'
config14='100  14  14  1024   256 1 1 0 0 1'
config15='100  7   7   1024  2048 1 1 0 0 1'
config16='100  7   7   1024   512 1 1 0 0 1'
config17='100  7   7   512   512 3 3 1 1 1'
config18='100  7   7   512  2048 1 1 0 0 1'
config19='100  7   7   2048   512 1 1 0 0 1'

GEMM_BLOCK=64
config_num=1
check_correctness=0
for config in "$config1" 
do
	for images in 1 28
	do
	        CONFIG_OUT=${config_num}_${images}_${OUT}
	        rm ${CONFIG_OUT}

		export OMP_NUM_THREADS=${images}
		for version in 0 1 2 3 4 5
		do
			params=( ${config} )
			ofw=${params[1]}
			ofh=${params[2]}
			nIfm=${params[3]}
			nOfm=${params[4]}
			#We will first do an actual run
			if [ $version -eq 0 -o $version -eq 1 ]
			then
				for (( T_oi=7; T_oi<= ${ofw}; T_oi=T_oi*2 ))
				do
				if [ `expr $ofw % $T_oi` -eq 0 ] 
				then
				for (( T_oj=7; T_oj<= ${ofh}; T_oj=T_oj*2 ))
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
                        		echo "${version}_${T_oi}_${T_oj}_${T_ifm_tile}_${T_ofm_tile},${GFLOPS}" >> ${CONFIG_OUT}
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
				echo "${version},${GFLOPS}" >> ${CONFIG_OUT}
			fi
		done
	done
	echo >> ${CONFIG_OUT}
	((config_num++))
done



