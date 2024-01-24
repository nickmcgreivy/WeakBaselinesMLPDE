# Weak baselines in machine learning for partial differential equation research

## System requirements

This code has been tested with MacOS. While we expect this code and instructions to work without modification on Linux, we provide no instructions for how to run this code on Windows. We have tested this code on Python v3.7 and v3.9. All required software can be downloaded using open-source package managers and compiled using binaries in this repository.

## Installation guide

Installation should not take more than a few minutes. First, use GitHub to clone this code by typing `git clone [HTTPS link here]` into command line. Type `cd WeakBaselinesMLPDE` to enter into the directory. Next, type
```
conda create -n pdeenv python=3.9
conda activate pdeenv
pip install pybind11 jax==0.4.13 jaxlib==0.4.13 sympy h5py scipy tree_math torch matplotlib jax-cfd xarray seaborn
```
Next, on mac run the following commands. On linux, replace `compilemac` and `compilemacLDLT` with `compilelinux` and `compilelinuxLDLT`. 
```
cd generate_sparse_solve
tar -xvf eigen-3.4.0.tar.bz2 
make compilemac
make compilemacLDLT
mv custom_call_* ..
cd ..
```
Finally, locate the path to the Anaconda source code. On my mac, this path is given by `/opt`. Once this path is located, type
```
CONDAPATH=[insert path here]
sed -i -e 's/norm(\[u.data for u in v\]/norm(jnp.asarray(\[u.data for u in v\])/g' $CONDAPATH/anaconda3/envs/pdeenv2/lib/python3.9/site-packages/jax_cfd/base/initial_conditions.py
```
Congratuations, you have compiled and installed all the necessary software to run all the code in this repository. Make sure to type `conda activate pdeenv` again if you close the command line terminal.

## Instructions for use

Each folder titled `article[insert number here]` contains code to reproduce one or more of the results in that article. Each folder contains instructions for running the code contained in that folder, as well as explanations for the output of that code. Most folders take only a few seconds, and no more than a minute, to run. Articles 4, 6, and 43 take minutes and/or hours to run, so we include a .png file with the expected output for each article.
