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
#include <iostream>

namespace {
  using std::experimental::basic_mdspan;
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::layout_left;
  using std::experimental::matrix_product;
  using std::experimental::transpose_view;
  using std::cout;
  using std::endl;

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
    constexpr size_t storageSize(7*maxDim*maxDim);
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
          offset += C_numRows * C_numCols;
          matrix_t A_t(storage.data() + offset, A_numCols, A_numRows);
          offset += A_numCols * A_numRows;
          matrix_t B_t(storage.data() + offset, B_numCols, B_numRows);
          offset += B_numCols * B_numRows;
          matrix_t C3(storage.data() + offset, C_numRows, C_numCols);
          offset += C_numRows * C_numCols;

          fill_matrix(A, scalar_t(real_t(1)));
          fill_matrix(B, scalar_t(real_t(2)));
          for (ptrdiff_t j = 0; j < A_numCols; ++j) {
            for (ptrdiff_t i = 0; i < A_numRows; ++i) {
              A_t(j,i) = A(i,j);
            }
          }
          for (ptrdiff_t j = 0; j < B_numCols; ++j) {
            for (ptrdiff_t i = 0; i < B_numRows; ++i) {
              B_t(j,i) = B(i,j);
            }
          }

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

          cout << " Test C2(" << C_numRows << " x " << C_numCols
               << ") = A(" << A_numRows << " x " << A_numCols
               << ") * B(" << B_numRows << " x " << B_numCols
               << ")" << endl;
          matrix_product(A, B, C2);
          {
            bool bad = false;
            for (ptrdiff_t j = 0; j < C_numCols; ++j) {
              for (ptrdiff_t i = 0; i < C_numRows; ++i) {
                if (C(i,j) != C2(i,j)) {
                  bad = true;
                  cout << " *** C(" << i << "," << j << ")=" << C(i,j)
                       << " != C2(" << i << "," << j << ")=" << C2(i,j)
                       << endl;
                }
              }
            }
          }

          // Fill result matrix with flag values to make sure that
          // we computed everything.
          for (ptrdiff_t j = 0; j < C_numCols; ++j) {
            for (ptrdiff_t i = 0; i < C_numRows; ++i) {
              C3(i,j) = std::numeric_limits<scalar_t>::min();
            }
          }

          cout << " Test C3(" << C3.extent(0) << " x " << C3.extent(1)
               << ") = "
               << "A_t(" << A_t.extent(0) << " x " << A_t.extent(1)
               << ")^T * "
               << "B_t(" << B_t.extent(0) << " x " << B_t.extent(1)
               << ")^T" << endl;
          matrix_product(transpose_view(A_t),
                         transpose_view(B_t), C3);
          {
            bool bad = false;
            for (ptrdiff_t j = 0; j < C_numCols; ++j) {
              for (ptrdiff_t i = 0; i < C_numRows; ++i) {
                if (C(i,j) != C3(i,j)) {
                  bad = true;
                  cout << " *** C(" << i << "," << j << ")=" << C(i,j)
                       << " != C3(" << i << "," << j << ")=" << C3(i,j)
                       << endl;
                }
              }
            }
          }
          {
            bool bad_transpose_view = false;

            auto A_tt = transpose_view(A_t);
            auto B_tt = transpose_view(B_t);
            if (A_tt.extent(0) != A.extent(0)) {
              cout << " *** A_tt.extent(0)=" << A_tt.extent(0)
                   << " != A.extent(0)=" << A.extent(0)
                   << endl;
              bad_transpose_view = true;
            }
            if (A_t.extent(0) != A.extent(1)) {
              cout << " *** A_t.extent(0)=" << A_t.extent(0)
                   << " != A.extent(1)=" << A.extent(1)
                   << endl;
              bad_transpose_view = true;
            }
            if (A_t.extent(1) != A.extent(0)) {
              cout << " *** A_t.extent(1)=" << A_t.extent(1)
                   << " != A.extent(0)=" << A.extent(0)
                   << endl;
              bad_transpose_view = true;
            }
            if (B_tt.extent(0) != B.extent(0)) {
              cout << " *** B_tt.extent(0)=" << B_tt.extent(0)
                   << " != B.extent(0)=" << B.extent(0)
                   << endl;
              bad_transpose_view = true;
            }
            if (B_t.extent(0) != B.extent(1)) {
              cout << " *** B_t.extent(0)=" << B_t.extent(0)
                   << " != B.extent(1)=" << B.extent(1)
                   << endl;
              bad_transpose_view = true;
            }
            if (B_t.extent(1) != B.extent(0)) {
              cout << " *** B_t.extent(1)=" << B_t.extent(1)
                   << " != B.extent(0)=" << B.extent(0)
                   << endl;
              bad_transpose_view = true;
            }
            if (A_tt.extent(1) != B_tt.extent(0)) {
              cout << " *** A_tt.extent(1)=" << A_tt.extent(1)
                   << " != B_tt.extent(0)=" << B_tt.extent(0)
                   << endl;
              bad_transpose_view = true;
            }
            if (bad_transpose_view) {
              return false;
            }

            for (ptrdiff_t j = 0; j < C_numCols; ++j) {
              for (ptrdiff_t i = 0; i < C_numRows; ++i) {
                C(i,j) = scalar_t(0.0); // this works even for complex
                for (ptrdiff_t k = 0; k < A_tt.extent(1); ++k) {
                  C(i,j) += A_tt(i,k) * B_tt(k,j);
                }
              }
            }
          }

          cout << " Compare using hand-rolled transpose_view loop"
               << endl;
          {
            bool bad = false;
            for (ptrdiff_t j = 0; j < C_numCols; ++j) {
              for (ptrdiff_t i = 0; i < C_numRows; ++i) {
                if (C(i,j) != C3(i,j)) {
                  bad = true;
                  cout << "C(" << i << "," << j << ")=" << C(i,j)
                       << " != C3(" << i << "," << j << ")=" << C3(i,j)
                       << endl;
                }
              }
            }
          }

          cout << " Test C3(" << C3.extent(0) << " x " << C3.extent(1)
               << ") = "
               << "2*A_t(" << A_t.extent(0) << " x " << A_t.extent(1)
               << ")^T * "
               << "B_t(" << B_t.extent(0) << " x " << B_t.extent(1)
               << ")^T" << endl;
          matrix_product(scaled_view(scalar_t(2.0), transpose_view(A_t)),
                         transpose_view(B_t), C3);
          {
            bool bad = false;
            for (ptrdiff_t j = 0; j < C_numCols; ++j) {
              for (ptrdiff_t i = 0; i < C_numRows; ++i) {
                if (scalar_t(2.0)*C(i,j) != C3(i,j)) {
                  bad = true;
                  cout << " *** 2*C(" << i << "," << j << ")=" << scalar_t(2.0)*C(i,j)
                       << " != C3(" << i << "," << j << ")=" << C3(i,j)
                       << endl;
                }
              }
            }
          }
          
        } // A_numCols
      } // C_numCols
    } // C_numRows
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
