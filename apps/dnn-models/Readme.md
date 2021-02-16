PolyDL_DNN_v1.ipynb ->
  Version used in paper with Original dataset.

PolyDL_DNN_v2.ipynb ->
  Improved version with seed correction of Neural Network on Original dataset.

PolyDL_DNN_v3_matmul.ipynb ->
  Improved version with Normalizing generated dataset during prediction phase.

PolyDL_DNN_v3_1_matmul_Total_zero.ipynb ->
  Version with MemDataSetSize set to 0.

matmul directory ->
  Upload all the data in this folder required to run on DNN model.

results_matmul ->
  All the results generated using dataset in matmul directory is added into this folder.

Command for execution -> 
  jupyter nbconvert --to notebook --execute PolyDL_DNN_v3_1_matmul_Total_zero.ipynb 
