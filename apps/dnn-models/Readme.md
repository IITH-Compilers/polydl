### Pre-requisite
     
1. Keras 2.2.4/tensorflow 1.14.0
     
     * ``Version used in local machine.``

2. mkdir matmul 
     
     * `` Create a Directory named "matmul" and store the sample.csv file in it.``

3. mkdir results_matmul 
     
     * `` Create a empty Directory named "results_matmul" and python script will generate the "results_sample.csv" and stored in this directory".``

### File Description


1. matmul directory 
     
     * ``Upload all the data in this folder required to run on DNN model.``

2. results_matmul
     
     * ``All the results generated using dataset in matmul directory is added into this folder.``

3. PolyDL_DNN_v1.ipynb
     
     * ``Version used in paper with Original dataset.``

4. PolyDL_DNN_v2.ipynb 
     
     * `` Improved version with seed correction of Neural Network on Original dataset.``

5. PolyDL_DNN_v3_matmul.ipynb 
     
     * ``Improved version with Normalizing generated dataset during prediction phase.``

6. PolyDL_DNN_v3_1_matmul_Total_zero.ipynb 
     
     * ``Version with MemDataSetSize set to 0.``

7. PolyDL_DNN_v4_matmul_SC_normalized.py 
     
     * ``Support functions to preprocess the data (as per SC Paper need with Unrolling factors) and added standard MinMaxScaler Normalisation .``

  

### Command for execution
  
1. jupyter nbconvert --to notebook --execute PolyDL_DNN_v3_1_matmul_Total_zero.ipynb 
