#include <experimental/linalg>

#include <iostream>

// Make mdspan less verbose
using std::experimental::mdspan;
using std::experimental::extents;
using std::dynamic_extent;
using std::experimental::submdspan;
using std::full_extent;

int main(int argc, char* argv[]) {
  std::cout << "Matrix Vector Product MixedPrec" << std::endl;
  int M = 40;
  {
    // Create Data
    std::vector<float> A_vec(M*8*4);
    std::vector<double> x_vec(M*4);
    std::vector<double> y_vec(M*8);

    // Create and initialize mdspan
    mdspan<float, extents<std::size_t, dynamic_extent, 8,4>> A(A_vec.data(),M);
    mdspan<double, extents<std::size_t, 4, dynamic_extent>> x(x_vec.data(),M);
    mdspan<double, extents<std::size_t, dynamic_extent, 8>> y(y_vec.data(),M);
    for(int m=0; m<A.extent(0); m++)
      for(int i=0; i<A.extent(1); i++)
        for(int j=0; j<A.extent(2); j++)
        A(m,i,j) = 1000.0 * m + 100.0 * i + j;
    for(int i=0; i<x.extent(0); i++)
      for(int m=0; m<x.extent(1); m++)
        x(i,m) = 33. * i + 0.33 * m;
    for(int m=0; m<y.extent(0); m++)
      for(int i=0; i<y.extent(1); i++)
        y(m,i) = 33. * m + 0.33 * i;

    for(int m = 0; m < M; m++) {
      auto A_m = submdspan(A, m, full_extent, full_extent);
      auto x_m = submdspan(x, full_extent, m);
      auto y_m = submdspan(y, m, full_extent);
      // y_m = A * x_m
      std::experimental::linalg::matrix_vector_product(A_m, x_m, y_m);
    }

    for(int i=0; i<y.extent(0); i+=5) std::cout << i << " " << y(i,1) << std::endl;
  }
}

