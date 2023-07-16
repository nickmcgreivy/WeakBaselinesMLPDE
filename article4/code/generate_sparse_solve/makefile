compilemac:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve.cpp -o custom_call_sparse_solve`python3-config --extension-suffix`

compilemacLDLT:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve_LDLT.cpp -o custom_call_sparse_solve_ldlt`python3-config --extension-suffix`


compilelinux:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve.cpp -o custom_call_sparse_solve`python3-config --extension-suffix`

compilelinuxLDLT:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve_LDLT.cpp -o custom_call_sparse_solve_ldlt`python3-config --extension-suffix`



clean:
