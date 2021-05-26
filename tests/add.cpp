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
  using std::experimental::linalg::add;

  TEST(BLAS1_add, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    constexpr ptrdiff_t storageSize = ptrdiff_t(3) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);
    vector_t z(storage.data() + 2*vectorSize, vectorSize);

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
      z(k) = 0.0;
    }

    add(x, y, z);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), x_k );
      EXPECT_EQ( y(k), y_k );
      EXPECT_EQ( z(k), x_k + y_k ); // check the output
    }
  }

  TEST(BLAS1_add, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    constexpr ptrdiff_t storageSize = ptrdiff_t(3) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);
    vector_t z(storage.data() + 2*vectorSize, vectorSize);

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      x(k) = x_k;
      y(k) = y_k;
      z(k) = scalar_t(0.0, 0.0);
    }

    add(x, y, z);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), x_k );
      EXPECT_EQ( y(k), y_k );
      EXPECT_EQ( z(k), x_k + y_k ); // check the output
    }
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
