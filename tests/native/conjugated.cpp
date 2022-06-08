#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::conjugated;

  TEST(conjugated, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize (5);
    constexpr std::size_t storageSize = std::size_t (2) * vectorSize;
    std::vector<scalar_t> storage (storageSize);

    vector_t x (storage.data (), vectorSize);
    vector_t y (storage.data () + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      const scalar_t y_k(real_t(k) + 2.0, real_t(k) + 2.0);
      x(k) = x_k;
      y(k) = y_k;
    }

    // Make sure that accessor_conjugate compiles
    {
      using accessor_t = vector_t::accessor_type;
      accessor_t accessor0;
      accessor_t accessor1 (y.accessor ());
      using std::experimental::linalg::accessor_conjugate;
      using accessor_conj_t = accessor_conjugate<accessor_t, scalar_t>;
      accessor_conj_t accessor2;
      accessor_conj_t accessor3 (y.accessor ());
    }

    auto y_conj = conjugated (y);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      EXPECT_EQ( x(k), x_k );

      // Make sure that conjugated doesn't modify the entries of
      // the original thing.
      const scalar_t y_k (real_t(k) + 2.0, real_t(k) + 2.0);
      EXPECT_EQ( y(k), y_k );

      const scalar_t y_k_conj (real_t(k) + 2.0, -real_t(k) - 2.0);
      EXPECT_EQ( y_conj(k), y_k_conj );
    }
  }
}
