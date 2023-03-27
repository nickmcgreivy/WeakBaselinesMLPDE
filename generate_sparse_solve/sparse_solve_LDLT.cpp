#include<Eigen/SparseCholesky>
#include<pybind11/eigen.h>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<iostream>
#include<cstdint>
#include <Eigen/OrderingMethods>
namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixXi;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::VectorX;

template <typename T>
void sparse_solve(void* out_ptr, void** data_ptr) {
	// Solves Mx=b where M is sparse
	T* b_ptr = reinterpret_cast<T *>(data_ptr[0]);    
  const int M_size = *reinterpret_cast<const int *>(data_ptr[3]);
  const bool forward = *reinterpret_cast<const bool *>(data_ptr[5]);

  static int prev_M_size = 0;
  //static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> forwardsolver;
  //static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> backwardsolver;

  static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> forwardsolver;
  static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> backwardsolver;
  if (M_size == prev_M_size) { 
  	// do nothing
  } else {

  	const std::int64_t nnz = *reinterpret_cast<const std::int64_t *>(data_ptr[4]);
  	T* sparse_data_ptr = reinterpret_cast<T *>(data_ptr[1]);
  	int* sparse_indices_ptr = reinterpret_cast<int *>(data_ptr[2]);
  	VectorX<T> M_data = Map<const VectorX<T>>(sparse_data_ptr, nnz);
  	MatrixXi M_indices = Map<const MatrixXi>(sparse_indices_ptr, nnz, 2);
	  // create matrix, create solver, and analyze it
	  std::vector<Eigen::Triplet<T>> tripletList;
	  tripletList.reserve(nnz);
	  for (int i = 0; i < nnz; ++i) {
	    tripletList.push_back(Eigen::Triplet<T>(M_indices(i,0),M_indices(i,1),M_data(i)));
	  }
	  Eigen::SparseMatrix<T> M(M_size,M_size);
	  M.setFromTriplets(tripletList.begin(), tripletList.end());

	  prev_M_size = M_size;

	  forwardsolver.analyzePattern(M);
		forwardsolver.factorize(M);
		backwardsolver.analyzePattern(M.transpose());
		backwardsolver.factorize(M.transpose());
	}

  VectorX<T> b = Map<const VectorX<T>>(b_ptr,M_size);
  T* x_ptr = reinterpret_cast<T *>(out_ptr);
  if (forward) {
  	Map<VectorX<T>>(x_ptr, M_size) = forwardsolver.solve(b);
  } else {
  	Map<VectorX<T>>(x_ptr, M_size) = backwardsolver.solve(b);
  }
}



template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["sparse_solve_f32"] = EncapsulateFunction(sparse_solve<float>);
  dict["sparse_solve_f64"] = EncapsulateFunction(sparse_solve<double>);
  return dict;
}
PYBIND11_MODULE(custom_call_sparse_solve_ldlt, m) { 
	m.def("registrations", &Registrations); 
}