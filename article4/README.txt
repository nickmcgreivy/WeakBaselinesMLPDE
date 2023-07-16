To run this code:

Step 1: create a new conda environment, then use pip to install the following packages:

pybind11
jax
jaxlib
sympy
h5py
scipy
tree_math
torch
matplotlib
jax-cfd

Step 2:

On mac, run the following commands. On linux, replace "compilemac" with "compilelinux". 

cd code/generate_sparse_solve
tar -xvf eigen-3.4.0.tar.bz2 
make compilemac
mv custom_call_* ..
cd ../..

Step 3: 

Replace the lines with TODO with in makefile.

Step 4:


To get FNO results, type 
'make print_FNO_statistics'


To get ML-accelerated CFD results, type 
'make compute_runtime'
'make compute_corrcoef'
'make plot_accuracy_runtime'
