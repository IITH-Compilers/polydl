CC=icc
CFLAGS=-O3 -I./libxsmm/include -qopenmp 
LDFLAGS=-L./libxsmm/lib -lxsmm -lxsmmext -lpthread

default: conv2d

./libxsmm/include/libxsmm.h:
	rm -rf libxsmm/
	git clone https://github.com/hfp/libxsmm.git
	$(MAKE) realclean -C libxsmm
	$(MAKE) BLAS=0 -C libxsmm

conv2d: conv2d.c ./libxsmm/include/libxsmm.h
	$(CC) $(CFLAGS) conv2d.c $(LDFLAGS) -o conv2d

clean: 
	rm -rf conv2d

