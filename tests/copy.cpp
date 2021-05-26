#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

#include <complex>
//#include <execution>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::linalg::copy;

  template<class Real>
  struct MakeVectorValues {
    static std::pair<Real, Real> make(const ptrdiff_t k) {
      const Real x_k = Real(k) + Real(1.0);
      const Real y_k = Real(k) + Real(2.0);
      return {x_k, y_k};
    }
  };

  template<class Real>
  struct MakeVectorValues<std::complex<Real>> {
    static std::pair<std::complex<Real>, std::complex<Real>>
    make (const ptrdiff_t k) {
      const std::complex<Real> x_k(Real(k) + 4.0, -Real(k) - 1.0);
      const std::complex<Real> y_k(Real(k) + 5.0, -Real(k) - 2.0);
      return {x_k, y_k};
    }
  };

  template<class Scalar>
  std::pair<Scalar, Scalar> makeVectorValues(const ptrdiff_t k) {
    return MakeVectorValues<Scalar>::make(k);
  }

  TEST(BLAS1_copy_vector, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    constexpr ptrdiff_t storageSize = ptrdiff_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      x(k) = vals.first;
      y(k) = vals.second;
    }

    copy(x, y);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), vals.first );
      EXPECT_EQ( y(k), vals.first ); // check the output
    }
  }

  TEST(BLAS1_copy_vector, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    constexpr ptrdiff_t storageSize = ptrdiff_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      x(k) = vals.first;
      y(k) = vals.second;
    }

    copy(x, y);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), vals.first );
      EXPECT_EQ( y(k), vals.first ); // check the output
    }
  }
}

template<class Real>
struct MakeMatrixValues {
  static std::pair<Real, Real>
  make(const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t numRows) {
    const Real A_ij = (Real(i) + Real(1.0)) +
      Real(numRows) * (Real(j) + Real(1.0));
    const Real B_ij = Real(i) + Real(2.0) +
      Real(numRows) * (Real(j) + Real(2.0));
    return {A_ij, B_ij};
  }
};

template<class Real>
struct MakeMatrixValues<std::complex<Real>> {
  static std::pair<std::complex<Real>, std::complex<Real>>
  make(const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t numRows) {
    using scalar_t = std::complex<Real>;
    const scalar_t A_i(Real(i) + 4.0, -Real(i) - 1.0);
    const scalar_t B_i(Real(i) + 5.0, -Real(i) - 2.0);
    const scalar_t A_j(Real(j) + 4.0, -Real(j) - 1.0);
    const scalar_t B_j(Real(j) + 5.0, -Real(j) - 2.0);

    const scalar_t A_ij = A_i + Real(numRows) * A_j;
    const scalar_t B_ij = B_i + Real(numRows) * B_j;
    return {A_ij, B_ij};
  }
};

template<class Scalar>
std::pair<Scalar, Scalar>
makeMatrixValues(const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t numRows) {
  return MakeMatrixValues<Scalar>::make(i, j, numRows);
}

TEST(BLAS1_copy_matrix, mdspan_double)
{
  using scalar_t = double;
  using matrix_t = basic_mdspan<scalar_t, extents<dynamic_extent, dynamic_extent>>;

  constexpr ptrdiff_t numRows(5);
  constexpr ptrdiff_t numCols(4);
  constexpr ptrdiff_t storageSize = ptrdiff_t(2) * ptrdiff_t(numRows*numCols);
  std::vector<scalar_t> storage(storageSize);

  matrix_t A(storage.data(), numRows, numCols);
  matrix_t B(storage.data() + numRows*numCols, numRows, numCols);

  for (ptrdiff_t j = 0; j < numCols; ++j) {
    for (ptrdiff_t i = 0; i < numRows; ++i) {
      const auto vals = makeMatrixValues<scalar_t>(i, j, numRows);
      A(i,j) = vals.first;
      B(i,j) = vals.second;
    }
  }

  copy(A, B);
  for (ptrdiff_t j = 0; j < numCols; ++j) {
    for (ptrdiff_t i = 0; i < numRows; ++i) {
      const auto vals = makeMatrixValues<scalar_t>(i, j, numRows);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( A(i,j), vals.first );
      EXPECT_EQ( B(i,j), vals.first ); // check the output
    }
  }
}

TEST(BLAS1_copy_matrix, mdspan_complex_double)
{
  using real_t = double;
  using scalar_t = std::complex<real_t>;
  using matrix_t = basic_mdspan<scalar_t, extents<dynamic_extent, dynamic_extent>>;

  constexpr ptrdiff_t numRows(5);
  constexpr ptrdiff_t numCols(4);
  constexpr ptrdiff_t storageSize = ptrdiff_t(2) * ptrdiff_t(numRows*numCols);
  std::vector<scalar_t> storage(storageSize);

  matrix_t A(storage.data(), numRows, numCols);
  matrix_t B(storage.data() + numRows*numCols, numRows, numCols);

  for (ptrdiff_t j = 0; j < numCols; ++j) {
    for (ptrdiff_t i = 0; i < numRows; ++i) {
      const auto vals = makeMatrixValues<scalar_t>(i, j, numRows);
      A(i,j) = vals.first;
      B(i,j) = vals.second;
    }
  }

  copy(A, B);
  for (ptrdiff_t j = 0; j < numCols; ++j) {
    for (ptrdiff_t i = 0; i < numRows; ++i) {
      const auto vals = makeMatrixValues<scalar_t>(i, j, numRows);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( A(i,j), vals.first );
      EXPECT_EQ( B(i,j), vals.first ); // check the output
    }
  }
}
