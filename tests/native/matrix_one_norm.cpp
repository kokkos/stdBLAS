#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <iostream>
#include <limits>
#include <vector>

namespace {
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using std::experimental::linalg::matrix_one_norm;
  using std::cout;
  using std::endl;

  template<class ElementType, class Layout>
  using basic_matrix_t = mdspan<
    ElementType,
    extents<
      std::size_t,
      dynamic_extent,
      dynamic_extent>,
    Layout,
    std::experimental::default_accessor<ElementType>>;

  template<class Scalar>
  struct Magnitude {
    using type = Scalar;
  };
  template<class Real>
  struct Magnitude<std::complex<Real>> {
    using type = Real;
  };

  template<class Scalar>
  using real_t = typename Magnitude<Scalar>::type;

  // Returns the one-norm of the matrix, after filling it.
  //
  // TODO add requires clause to constrain Layout to be unique.
  template<class ElementType, class Layout>
  real_t<ElementType>
  fill_matrix(basic_matrix_t<ElementType, Layout> A, const ElementType startVal)
  {
    using std::abs;
    using std::max;
    const std::size_t A_numRows = A.extent(0);
    const std::size_t A_numCols = A.extent(1);

    real_t<ElementType> maxColNorm{};
    for (std::size_t j = 0; j < A_numCols; ++j) {
      real_t<ElementType> curColOneNorm{};
      for (std::size_t i = 0; i < A_numRows; ++i) {
        const auto A_ij = (ElementType(i)+startVal) +
          (ElementType(j)+startVal) * ElementType(A_numRows);
        A(i,j) = A_ij;
        curColOneNorm += abs(A_ij);
      }
      maxColNorm = std::max(maxColNorm, curColOneNorm);
    }
    return maxColNorm;
  }

  template<class Scalar>
  void test_matrix_one_norm()
  {
    using std::abs;
    using scalar_t = Scalar;
    using std::experimental::layout_left;
    using matrix_t = basic_matrix_t<scalar_t, layout_left>;

    constexpr size_t maxNumRows = 7;
    constexpr size_t maxNumCols = 7;
    constexpr size_t storageSize(maxNumRows * maxNumCols);
    std::vector<scalar_t> storage(storageSize);
    for (size_t A_numRows : {0, 1, 4, 7}) {
      for (size_t A_numCols : {0, 1, 4, 7}) {
        // NOTE (Hoemmen 2021/05/26) Need the cast to ptrdiff_t
        // to avoid a possible bug in the current mdspan implementation,
        // and/or an MSVC bug.  I'll need to check more recent MSVC versions;
        // I'm building right now with MSVC 2019 16.7.0.
        matrix_t A(storage.data(), std::size_t(A_numRows), std::size_t(A_numCols));

        const auto startVal = scalar_t(real_t<scalar_t>(1.0));
        const real_t<scalar_t> expectedResult = fill_matrix(A, startVal);
        const real_t<scalar_t> maxMatrixValueAbs = real_t<scalar_t>(A_numRows * A_numCols) + abs(startVal);
        const real_t<scalar_t> computedTwoArgResult = matrix_one_norm(A, real_t<scalar_t>{});
        const real_t<scalar_t> computedOneArgResult = matrix_one_norm(A);
        cout << "Computed matrix_one_norm(2 args): " << computedTwoArgResult << endl
             << "Computed matrix_one_norm(1 arg): " << computedOneArgResult << endl
             << "Expected matrix_one_norm: " << expectedResult << endl;
        if constexpr (std::is_floating_point_v<real_t<scalar_t>>) {
          // Matrix one-norm tolerance depends only on the number of rows,
          // since the only operations that might round are the column sums.
          //
          // Use the max abs matrix element as a multiplier.
          const auto multiplier = maxMatrixValueAbs < abs(startVal) ? abs(startVal) : maxMatrixValueAbs;
          const real_t<scalar_t> tolerance = multiplier * A_numRows * std::numeric_limits<real_t<scalar_t>>::epsilon();
          EXPECT_NEAR( computedTwoArgResult, expectedResult, tolerance );
          EXPECT_NEAR( computedOneArgResult, expectedResult, tolerance );
        } else {
          EXPECT_EQ( computedTwoArgResult, expectedResult );
          EXPECT_EQ( computedOneArgResult, expectedResult );
        }
      }
    }
  }

  TEST(matrix_one_norm, mdspan_int)
  {
    test_matrix_one_norm<int>();
  }

  TEST(matrix_one_norm, mdspan_double)
  {
    test_matrix_one_norm<double>();
  }

  TEST(matrix_one_norm, mdspan_complex_float)
  {
    test_matrix_one_norm<std::complex<float>>();
  }
}
