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
  using std::experimental::basic_mdspan;
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::layout_left;
  using std::experimental::matrix_product;

  template<class MdspanType, class Scalar>
  struct FillMatrix {
    static void fill(MdspanType A, const Scalar startVal);
  };

  template<class MdspanType>
  struct FillMatrix<MdspanType, int> {
    static void fill(MdspanType A, const int startVal)
    {
      const ptrdiff_t A_numRows = A.extent(0);
      const ptrdiff_t A_numCols = A.extent(1);
      for (ptrdiff_t j = 0; j < A_numCols; ++j) {
        for (ptrdiff_t i = 0; i < A_numRows; ++i) {
          A(i,j) = int((i+startVal) + (j+startVal) * A_numRows);
        }
      }
    }
  };

  template<class MdspanType>
  struct FillMatrix<MdspanType, double> {
    static void fill(MdspanType A, const double startVal)
    {
      const ptrdiff_t A_numRows = A.extent(0);
      const ptrdiff_t A_numCols = A.extent(1);
      for (ptrdiff_t j = 0; j < A_numCols; ++j) {
        for (ptrdiff_t i = 0; i < A_numRows; ++i) {
          A(i,j) = (double(i)+startVal) +
            (double(j)+startVal) * double(A_numRows);
        }
      }
    }
  };

  template<class MdspanType, class Scalar>
  void fill_matrix(MdspanType A, const Scalar startVal) {
    FillMatrix<MdspanType, Scalar>::fill(A, startVal);
  }

  template<class Scalar>
  struct Magnitude {
    using type = Scalar;
  };
  template<class Real>
  struct Magnitude<std::complex<Real>> {
    using type = Real;
  };

  template<class Scalar>
  bool test_matrix_product()
  {
    using scalar_t = Scalar;
    using real_t = typename Magnitude<Scalar>::type;

    using extents_t = extents<dynamic_extent, dynamic_extent>;
    using matrix_t = basic_mdspan<scalar_t, extents_t, layout_left>;

    constexpr size_t maxDim = 7;
    constexpr size_t storageSize(4*maxDim*maxDim);
    std::vector<scalar_t> storage(storageSize);

    for (ptrdiff_t C_numRows : {1, 4, 7}) {
      for (ptrdiff_t C_numCols : {1, 4, 7}) {
        for (ptrdiff_t A_numCols : {1, 4, 7}) {

          const ptrdiff_t A_numRows = C_numRows;
          const ptrdiff_t B_numRows = A_numCols;
          const ptrdiff_t B_numCols = C_numCols;

          ptrdiff_t offset = 0;
          matrix_t A(storage.data() + offset, A_numRows, A_numCols);
          offset += A_numRows * A_numCols;
          matrix_t B(storage.data() + offset, B_numRows, B_numCols);
          offset += B_numRows * B_numCols;
          matrix_t C(storage.data() + offset, C_numRows, C_numCols);
          offset += C_numRows * C_numCols;
          matrix_t C2(storage.data() + offset, C_numRows, C_numCols);

          fill_matrix(A, scalar_t(real_t(1)));
          fill_matrix(B, scalar_t(real_t(2)));

          for (ptrdiff_t j = 0; j < C_numCols; ++j) {
            for (ptrdiff_t i = 0; i < C_numRows; ++i) {
              C(i,j) = scalar_t(0.0); // this works even for complex
              for (ptrdiff_t k = 0; k < A_numCols; ++k) {
                C(i,j) += A(i,k) * B(k,j);
              }
            }
          }

          // Fill result matrix with flag values to make sure that we
          // computed everything.
          for (ptrdiff_t j = 0; j < C_numCols; ++j) {
            for (ptrdiff_t i = 0; i < C_numRows; ++i) {
              C2(i,j) = std::numeric_limits<scalar_t>::min();
            }
          }

          matrix_product(A, B, C2);

          for (ptrdiff_t j = 0; j < C_numCols; ++j) {
            for (ptrdiff_t i = 0; i < C_numRows; ++i) {
              if (C(i,j) != C2(i,j)) {
                return false;
              }
            }
          }
        }
      }
    }
    return true;
  }

  // Testing int is a way to test the non-BLAS-library implementation.
  TEST(BLAS3_gemm, mdspan_int)
  {
    const bool result = test_matrix_product<int>();
    EXPECT_TRUE( result );
  }

  TEST(BLAS3_gemm, mdspan_double)
  {
    const bool result = test_matrix_product<double>();
    EXPECT_TRUE( result );
  }
}
