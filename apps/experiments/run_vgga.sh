
FILE=$1
ITERS=1000 
#vgga
 sh ./$FILE ${ITERS} 224 224       3   64  3  3 1 1 1  vgga_1 #
 sh ./$FILE ${ITERS} 112 112      64  128  3  3 1 1 1  vgga_2
 sh ./$FILE ${ITERS}  56  56     128  256  3  3 1 1 1  vgga_3
 sh ./$FILE ${ITERS}  56  56     256  256  3  3 1 1 1  vgga_4
 sh ./$FILE ${ITERS}  28  28     256  512  3  3 1 1 1  vgga_5
 sh ./$FILE ${ITERS}  28  28     512  512  3  3 1 1 1  vgga_6
 sh ./$FILE ${ITERS}  14  14     512  512  3  3 1 1 1  vgga_7
