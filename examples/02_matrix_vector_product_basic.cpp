#include <experimental/linalg>

#include <iostream>

// Make mdspan less verbose
using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;

int main(int argc, char* argv[]) {
  std::cout << "Matrix Vector Product Basic" << std::endl;
  int N = 40, M = 20;
  {
    // Create Data
    std::vector<double> A_vec(N*M);
    std::vector<double> x_vec(M);
    std::vector<double> y_vec(N);

    // Create and initialize mdspan
    // Would look simple with CTAD, GCC 11.1 works but some others are buggy
    mdspan<double, extents<std::size_t, dynamic_extent,dynamic_extent>> A(A_vec.data(),N,M);
    mdspan<double, extents<std::size_t, dynamic_extent>> x(x_vec.data(),M);
    mdspan<double, extents<std::size_t, dynamic_extent>> y(y_vec.data(),N);
    for(int i=0; i<A.extent(0); i++)
      for(int j=0; j<A.extent(1); j++)
        A(i,j) = 100.0*i+j;
    for(int i=0; i<x.extent(0); i++)
      x(i) = 1. * i;
    for(int i=0; i<y.extent(0); i++)
      y(i) = -1. * i;

    // y = A * x
    std::experimental::linalg::matrix_vector_product(A, x, y);

    // y = 0.5 * y + 2 * A * x
#ifdef LINALG_HAS_EXECUTION
    std::experimental::linalg::matrix_vector_product(std::execution::par,
       std::experimental::linalg::scaled(2.0, A), x,
       std::experimental::linalg::scaled(0.5, y), y);
#else
    std::experimental::linalg::matrix_vector_product(
       std::experimental::linalg::scaled(2.0, A), x,
       std::experimental::linalg::scaled(0.5, y), y);
#endif
    for(int i=0; i<y.extent(0); i+=5) std::cout << i << " " << y(i) << std::endl;
  }
}
