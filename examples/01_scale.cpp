// Examples currently use parentheses (e.g., A(i,j))
// for the array access operator,
// instead of square brackets (e.g., A[i,j]).
// This must be defined before including any mdspan headers.
#define MDSPAN_USE_PAREN_OPERATOR 1

#include <mdspan/mdspan.hpp>
#include "experimental/__p2630_bits/submdspan.hpp"
#include <experimental/linalg>
#include <iostream>
#include <vector>

#ifdef LINALG_HAS_EXECUTION
#  include <execution>
#endif

namespace MdSpan = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace LinearAlgebra = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: linalg;

using MdSpan::mdspan;
using MdSpan::extents;
#if defined(__cpp_lib_span)
#include <span>
  using std::dynamic_extent;
#else
  using MdSpan::dynamic_extent;
#endif

int main(int argc, char* argv[]) {
  std::cout << "Scale" << std::endl;
  int N = 40;
  {
    // Create Data
    std::vector<double> x_vec(N);

    // Create and initialize mdspan
    //
    // With CTAD working we could do the following.
    // GCC 11.1 works but some other compilers are buggy.
    //
    // mdspan x(x_vec.data(), N);
    mdspan<double, extents<int, dynamic_extent>> x(x_vec.data(), N);
    for (int i = 0; i < x.extent(0); ++i) {
      x(i) = i;
    }

    // Call linalg::scale x = 2.0*x;
    LinearAlgebra::scale(2.0, x);
#ifdef LINALG_HAS_EXECUTION
    LinearAlgebra::scale(std::execution::par, 2.0, x);
#else
    LinearAlgebra::scale(2.0, x);
#endif

    for (int i = 0; i < x.extent(0); i += 5) {
      std::cout << i << " " << x(i) << std::endl;
    }
  }
}
