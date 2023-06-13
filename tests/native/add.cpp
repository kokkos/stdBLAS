#include "./gtest_fixtures.hpp"

#include <experimental/linalg>
#include <array>
#include <vector>

namespace {
  using std::experimental::linalg::add;

  TEST(BLAS1_add, vector_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

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

  TEST(BLAS1_add, vector_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

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

  TEST(BLAS1_add, matrix_double)
  {
    using scalar_t = double;
    constexpr std::size_t numRows(5);
    constexpr std::size_t numCols(6);
    constexpr std::size_t matrixSize = numRows * numCols;
    std::array<scalar_t, matrixSize> A_storage;
    std::array<scalar_t, matrixSize> B_storage;
    std::array<scalar_t, matrixSize> C_storage;

    using matrix_t = mdspan<scalar_t, extents<std::size_t, numRows, numCols>>;
    matrix_t A(A_storage.data());
    matrix_t B(B_storage.data());
    matrix_t C(C_storage.data());

    for (std::size_t c = 0; c < numCols; ++c) {
      for (std::size_t r = 0; r < numRows; ++r) {
	const scalar_t A_rc = scalar_t(c) + scalar_t(numCols) * scalar_t(r);
	const scalar_t B_rc = scalar_t(2.0) * A_rc;
	A(r,c) = A_rc;
	B(r,c) = B_rc;
	C(r,c) = scalar_t{};
      }
    }
    add(A, B, C);
    for (std::size_t c = 0; c < numCols; ++c) {
      for (std::size_t r = 0; r < numRows; ++r) {
	const scalar_t A_rc = scalar_t(c) + scalar_t(numCols) * scalar_t(r);
	const scalar_t B_rc = scalar_t(2.0) * A_rc;
	// Make sure the function didn't modify the input.
	EXPECT_EQ( A(r,c), A_rc );
	EXPECT_EQ( B(r,c), B_rc );
	EXPECT_EQ( C(r,c), A_rc + B_rc ); // check the output
      }
    }
  }
}
