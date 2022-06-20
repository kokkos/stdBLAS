#include <experimental/linalg>

#include <iostream>

#if (! defined(__GNUC__)) || (__GNUC__ > 9)
#  define MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES 1
#endif

#ifdef MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES
#  include <execution>
#endif

// Make mdspan less verbose
using std::experimental::mdspan;
using std::experimental::extents;
using std::experimental::dynamic_extent;

int main(int argc, char* argv[]) {
  std::cout << "Scale" << std::endl;
  int N = 40;
  {
    // Create Data
    std::vector<double> x_vec(N);

    // Create and initialize mdspan
    // With CTAD working we could do, GCC 11.1 works but some others are buggy
    // mdspan x(x_vec.data(), N);
    mdspan<double, extents<dynamic_extent>> x(x_vec.data(),N);
    for(int i=0; i<x.extent(0); i++) x(i) = i;

    // Call linalg::scale x = 2.0*x;
    std::experimental::linalg::scale(2.0, x);
#ifdef MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES
    std::experimental::linalg::scale(std::execution::par, 2.0, x);
#else
    std::experimental::linalg::scale(2.0, x);
#endif

    for(int i=0; i<x.extent(0); i+=5) std::cout << i << " " << x(i) << std::endl;
  }
}
