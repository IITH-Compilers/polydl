ITERS=1000
#yolov2
# sh ./run_conv.sh ${ITERS} 610 610   3 32 3 3 0 0 1  yolov2_1
 sh ./run_conv.sh ${ITERS} 306 306   32 64 3 3 0 0 1  yolov2_2
 sh ./run_conv.sh ${ITERS} 154 154   64 128 3 3 0 0 1  yolov2_3
 sh ./run_conv.sh ${ITERS} 152 152   128 64 1 1 0 0 1  yolov2_4
 sh ./run_conv.sh ${ITERS} 78 78   128 256 3 3 0 0 1  yolov2_5
 sh ./run_conv.sh ${ITERS} 76 76   256 128 1 1 0 0 1  yolov2_6
 sh ./run_conv.sh ${ITERS} 40 40   256 512 3 3 0 0 1  yolov2_7
 sh ./run_conv.sh ${ITERS} 38 38   512 256 1 1 0 0 1  yolov2_8
 sh ./run_conv.sh ${ITERS} 21 21   512 1024 3 3 0 0 1  yolov2_9
 sh ./run_conv.sh ${ITERS} 19 19   1024 512 1 1 0 0 1  yolov2_10
 sh ./run_conv.sh ${ITERS} 21 21   1024 1024 3 3 0 0 1  yolov2_11
 sh ./run_conv.sh ${ITERS} 38 38   512 64 1 1 0 0 1  yolov2_12
 sh ./run_conv.sh ${ITERS} 21 21   1280 1024 3 3 0 0 1  yolov2_13
# sh ./run_conv.sh ${ITERS} 19 19   1024 425 1 1 0 0 1  yolov2_14
