#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>

namespace {
  using std::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::swap_elements;

  TEST(BLAS1_swap, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      x[k] = x_k;
      y[k] = y_k;
    }

    swap_elements(x, y);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      EXPECT_EQ( x[k], y_k );
      EXPECT_EQ( y[k], x_k );
    }
  }

  TEST(BLAS1_swap, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      x[k] = x_k;
      y[k] = y_k;
    }

    swap_elements(x, y);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      EXPECT_EQ( x[k], y_k );
      EXPECT_EQ( y[k], x_k );
    }
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
