### Pre-requisite
     
1. Keras 2.2.4/tensorflow 1.14.0
     
     * ``Version used in local machine.``

2. mkdir matmul 
     
     * `` Create a Directory named "matmul" and store the sample.csv file in it.``

3. mkdir results_matmul 
     
     * `` Create a empty Directory named "results_matmul" and python script will generate the "results_sample.csv" and stored in this directory".``

### File Description

1. PolyDL_DNN_v1.ipynb
     
     * ``Version used in paper with Original dataset.``

2. PolyDL_DNN_v2.ipynb 
     
     * `` Improved version with seed correction of Neural Network on Original dataset.``

3. PolyDL_DNN_v3_matmul.ipynb 
     
     * ``Improved version with Normalizing generated dataset during prediction phase.``

4. PolyDL_DNN_v3_1_matmul_Total_zero.ipynb 
     
     * ``Version with MemDataSetSize set to 0.``

5. matmul directory 
     
     * ``Upload all the data in this folder required to run on DNN model.``

6. results_matmul
     
     * ``All the results generated using dataset in matmul directory is added into this folder.``
  

### Command for execution
  
1. jupyter nbconvert --to notebook --execute PolyDL_DNN_v3_1_matmul_Total_zero.ipynb 
