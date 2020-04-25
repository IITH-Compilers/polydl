rm -rf mkl-dnn
git clone https://github.com/intel/mkl-dnn.git
cd mkl-dnn
git checkout tags/v1.4
mkdir -p build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install/ ..
make -j50
make doc
make install
