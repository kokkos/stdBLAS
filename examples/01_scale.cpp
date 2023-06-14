#include <experimental/linalg>

#include <iostream>

// Make mdspan less verbose
using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;

int main(int argc, char* argv[]) {
  std::cout << "Scale" << std::endl;
  int N = 40;
  {
    // Create Data
    std::vector<double> x_vec(N);

    // Create and initialize mdspan
    // With CTAD working we could do, GCC 11.1 works but some others are buggy
    // mdspan x(x_vec.data(), N);
    mdspan<double, extents<std::size_t, dynamic_extent>> x(x_vec.data(),N);
    for(int i=0; i<x.extent(0); i++) x(i) = i;

    // Call linalg::scale x = 2.0*x;
    std::experimental::linalg::scale(2.0, x);
#ifdef LINALG_HAS_EXECUTION
    std::experimental::linalg::scale(std::execution::par, 2.0, x);
#else
    std::experimental::linalg::scale(2.0, x);
#endif

    for(int i=0; i<x.extent(0); i+=5) std::cout << i << " " << x(i) << std::endl;
  }
}
