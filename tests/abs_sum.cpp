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
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::vector_abs_sum;

  TEST(BLAS1_abs_sum, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(11);
    std::vector<scalar_t> storage(vectorSize);

    vector_t x(storage.data(), vectorSize);
    scalar_t result = 0.0;

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;

      // Give the vector some negative entries to make sure abs is being used
      if (k % 2 == 0) {
        x(k) = -x_k;
      }
      else {
        x(k) = x_k;
      }
    }

    vector_abs_sum(x, result);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      scalar_t x_k = scalar_t (k) + 1.0;
      if (k % 2 == 0) {
        x_k = -x_k;
      }
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), x_k );
    }
    // check the output
    EXPECT_EQ( result, vectorSize*(vectorSize+1) / 2.0);
  }
}

