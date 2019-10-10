
ITERS=1000 
#vgga
# sh ./run_conv.sh ${ITERS} 224 224       3   64  3  3 1 1 1  vgga_1
 sh ./run_conv.sh ${ITERS} 112 112      64  128  3  3 1 1 1  vgga_2
 sh ./run_conv.sh ${ITERS}  56  56     128  256  3  3 1 1 1  vgga_3
 sh ./run_conv.sh ${ITERS}  56  56     256  256  3  3 1 1 1  vgga_4
 sh ./run_conv.sh ${ITERS}  28  28     256  512  3  3 1 1 1  vgga_5
 sh ./run_conv.sh ${ITERS}  28  28     512  512  3  3 1 1 1  vgga_6
 sh ./run_conv.sh ${ITERS}  14  14     512  512  3  3 1 1 1  vgga_7
