rm -rf perf.csv
cd mkl-dnn/build/
cp ../../run_bench.sh ./tests/benchdnn/
cd ./tests/benchdnn
sh run_bench.sh
cp ./perf.csv ../../../../
echo "success"
