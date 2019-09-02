#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

//#include <execution>
#include <vector>
#include "gtest/gtest.h"

namespace {
  TEST(BLAS1_dot, mdspan_double)
  {
    using std::experimental::dynamic_extent;
    using std::experimental::extents;
    using std::experimental::basic_mdspan;

    using vector_t = basic_mdspan<double, extents<dynamic_extent>>;

    constexpr size_t vectorSize (5);
    constexpr size_t storageSize = size_t (2) * vectorSize;
    std::vector<double> storage (storageSize);

    vector_t x (storage.data (), vectorSize);
    vector_t y (storage.data () + vectorSize, vectorSize);

    double expectedDotResult = 0.0;
    for (size_t k = 0; k < vectorSize; ++k) {
      const double x_k = double (k) + 1.0;
      const double y_k = double (k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
    }

    double dotResult = 0.0;
    std::experimental::dot (x, y, dotResult);
    EXPECT_EQ( dotResult, expectedDotResult );


    double dotResultPar = 0.0;
    // See note above.
    //std::experimental::dot (std::execution::par, x, y, dotResultPar);

    // This is noncomforming, but I need some way to test the executor overloads.
    using fake_executor_t = int;
    std::experimental::dot (fake_executor_t (), x, y, dotResultPar);
    EXPECT_EQ( dotResultPar, expectedDotResult );
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
