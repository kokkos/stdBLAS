#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::add;

  TEST(BLAS1_add, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(3) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);
    vector_t z(storage.data() + 2*vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
      z(k) = 0.0;
    }

    add(x, y, z);
    for (std::size_t k = 0; k < vectorSize; ++k) {
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
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(3) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);
    vector_t z(storage.data() + 2*vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      x(k) = x_k;
      y(k) = y_k;
      z(k) = scalar_t(0.0, 0.0);
    }

    add(x, y, z);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), x_k );
      EXPECT_EQ( y(k), y_k );
      EXPECT_EQ( z(k), x_k + y_k ); // check the output
    }
  }
}
