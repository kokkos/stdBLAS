#include <experimental/linalg>
#include <experimental/mdspan>

//#include <execution> // TODO (Hoemmen 2021/05/26) test these overloads
#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include <iostream>

namespace {
  //using std::experimental::basic_mdspan;
  //using std::experimental::dynamic_extent;
  //using std::experimental::extents;
  using std::experimental::linalg::matrix_one_norm;
  using std::cout;
  using std::endl;
  
  template<class ElementType, class Layout>
  using basic_matrix_t = std::experimental::basic_mdspan<
    ElementType, 
    std::experimental::extents<
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent>,
    Layout,
    std::experimental::accessor_basic<ElementType>>;

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
        matrix_t A(storage.data(), ptrdiff_t(A_numRows), ptrdiff_t(A_numCols));

        const real_t<scalar_t> expectedResult = fill_matrix(A, scalar_t(real_t<scalar_t>(1)));
        const real_t<scalar_t> computedResult = matrix_one_norm(A, real_t<scalar_t>{});
        cout << "Computed matrix_one_norm: " << computedResult << endl
             << "Expected matrix_one_norm: " << expectedResult << endl;
        if constexpr (std::is_floating_point_v<real_t<scalar_t>>) {
          // Matrix one-norm tolerance depends only on the number of rows,
          // since the only operations that might round are the column sums.
          const real_t<scalar_t> tolerance = 10.0 * A_numRows * std::numeric_limits<real_t<scalar_t>>::epsilon();
          EXPECT_NEAR( computedResult, expectedResult, tolerance );
        } else {
          EXPECT_EQ( computedResult, expectedResult );
        }
      }
    }
  }

  //TEST(matrix_one_norm, mdspan_int)
  //{
  //  test_matrix_one_norm<int>();
  //} 

  //TEST(matrix_one_norm, mdspan_unsigned_int)
  //{
  //  test_matrix_one_norm<unsigned int>();
  //}

  TEST(matrix_one_norm, mdspan_double)
  {
    test_matrix_one_norm<double>();
  }

  TEST(matrix_one_norm, mdspan_complex_float)
  {
    test_matrix_one_norm<std::complex<float>>();
  }
}
