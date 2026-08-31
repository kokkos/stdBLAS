#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::scale;

  TEST(BLAS1_scale, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        x(k) = x_k;
      }
      const scalar_t scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        x(k) = x_k;
      }
      const float scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
  }

  TEST(BLAS1_scale, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        x(k) = x_k;
      }
      const real_t scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        x(k) = x_k;
      }
      const scalar_t scaleFactor (5.0, -1.0);
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
  }

  // Stride other than 1. Both halves matter: the referenced elements must be
  // scaled and the ones between them must not, which is what a wrong incx gets
  // wrong.
  TEST(BLAS1_scale, mdspan_layout_stride)
  {
    using scalar_t = double;
    using extents_t = extents<std::size_t, dynamic_extent>;
    using vector_t = mdspan<scalar_t, extents_t, layout_stride>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t stride(2);
    std::vector<scalar_t> storage(vectorSize * stride);

    layout_stride::mapping<extents_t> mapping{
      extents_t{vectorSize}, std::array<std::size_t, 1>{stride}};
    vector_t x(storage.data(), mapping);

    for (std::size_t k = 0; k < storage.size(); ++k) {
      storage[k] = scalar_t (k) + 1.0;
    }
    const scalar_t scaleFactor = 5.0;
    scale(scaleFactor, x);
    for (std::size_t k = 0; k < storage.size(); ++k) {
      const scalar_t storage_k = scalar_t (k) + 1.0;
      const scalar_t expected =
        (k % stride == 0) ? scaleFactor * storage_k : storage_k;
      EXPECT_EQ( storage[k], expected );
    }
  }

  TEST(BLAS1_scale, mdspan_layout_left_padded)
  {
    using scalar_t = double;
    using extents_t = extents<std::size_t, 4>;
    using layout_t = layout_left_padded<4>;
    using vector_t = mdspan<scalar_t, extents_t, layout_t>;

    constexpr std::size_t vectorSize(4);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    layout_t::mapping<extents_t> mapping{extents_t{}};
    vector_t x(storage.data(), mapping);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      x(k) = x_k;
    }
    const scalar_t scaleFactor = 5.0;
    scale(scaleFactor, x);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      EXPECT_EQ( x(k), scaleFactor * x_k );
    }
  }

  TEST(BLAS1_scale, mdspan_layout_right_padded)
  {
    using scalar_t = double;
    using extents_t = extents<std::size_t, 4>;
    using layout_t = layout_right_padded<dynamic_extent>;
    using vector_t = mdspan<scalar_t, extents_t, layout_t>;

    constexpr std::size_t vectorSize(4);
    constexpr std::size_t paddingValue(4);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    layout_t::mapping<extents_t> mapping{extents_t{}, paddingValue};
    vector_t x(storage.data(), mapping);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      x(k) = x_k;
    }
    const scalar_t scaleFactor = 5.0;
    scale(scaleFactor, x);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      EXPECT_EQ( x(k), scaleFactor * x_k );
    }
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
