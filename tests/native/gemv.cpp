#define MDSPAN_USE_PAREN_OPERATOR 1

#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>
#include <iostream>

namespace {
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left;
  using std::experimental::linalg::matrix_vector_product;
  using std::experimental::linalg::transposed;
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
      const std::size_t A_numRows = A.extent(0);
      const std::size_t A_numCols = A.extent(1);
      for (std::size_t j = 0; j < A_numCols; ++j) {
        for (std::size_t i = 0; i < A_numRows; ++i) {
          A(i,j) = int((i+startVal) + (j+startVal) * A_numRows);
        }
      }
    }
  };

  template<class MdspanType>
  struct FillMatrix<MdspanType, double> {
    static void fill(MdspanType A, const double startVal)
    {
      const std::size_t A_numRows = A.extent(0);
      const std::size_t A_numCols = A.extent(1);
      for (std::size_t j = 0; j < A_numCols; ++j) {
        for (std::size_t i = 0; i < A_numRows; ++i) {
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

  template<class MdspanType, class Scalar>
  struct FillVector {
    static void fill(MdspanType x, const Scalar startVal);
  };

  template<class MdspanType>
  struct FillVector<MdspanType, int> {
    static void fill(MdspanType x, const int startVal)
    {
      const std::size_t numRows = x.extent(0);
      for (std::size_t i = 0; i < numRows; ++i) {
        x(i) = int(i+startVal);
      }
    }
  };

  template<class MdspanType>
  struct FillVector<MdspanType, double> {
    static void fill(MdspanType x, const double startVal)
    {
      const std::size_t numRows = x.extent(0);
      for (std::size_t i = 0; i < numRows; ++i) {
        x(i) = double(i+startVal);
      }
    }
  };

  template<class MdspanType, class Scalar>
  void fill_vector(MdspanType x, const Scalar startVal) {
    FillVector<MdspanType, Scalar>::fill(x, startVal);
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
  void test_matrix_product()
  {
    using scalar_t = Scalar;
    using real_t = typename Magnitude<Scalar>::type;

    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using matrix_t = mdspan<scalar_t, extents_t, layout_left>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>, layout_left>;

    constexpr std::size_t maxDim = 7;
    constexpr std::size_t storageSize(maxDim*maxDim + 3*maxDim);
    std::vector<scalar_t> storage(storageSize);

    for (std::size_t numRows : {1, 4, 7}) {
      for (std::size_t numCols : {1, 4, 7}) {
        std::size_t offset = 0;
        matrix_t A(storage.data() + offset, numRows, numCols);
        offset += numRows * numCols;
        vector_t x(storage.data() + offset, numCols);
        offset += numCols;
        vector_t y(storage.data() + offset, numRows);
        offset += numRows;
        vector_t gs(storage.data() + offset, numRows);

        fill_matrix(A, scalar_t(real_t(1)));
        fill_vector(x, scalar_t(real_t(2)));

        // Initialize vector to zero
        for (std::size_t i = 0; i < numRows; i++) {
          gs(i) = scalar_t(0.0); // this works even for complex
        }

        // Perform matvec
        for (std::size_t j = 0; j < numCols; ++j) {
          for (std::size_t i = 0; i < numRows; ++i) {
            gs(i) += A(i,j) * x(j);
          }
        }

        // Fill result vector with flag values to make sure that we
        // computed everything.
        for (std::size_t j = 0; j < numRows; ++j) {
          y(j) = std::numeric_limits<scalar_t>::min();
        }

        cout << " Test y = A(" << numRows << " x " << numCols
             << ") * x = y\n";
        matrix_vector_product(A, x, y);
        for (std::size_t i = 0; i < numRows; ++i) {
          EXPECT_DOUBLE_EQ(y(i), gs(i)) << "Vectors differ at index " << i;
        }
      } // numCols
    } // numRows
  }

  // Testing int is a way to test the non-BLAS-library implementation.
  TEST(BLAS3_gemm, mdspan_int)
  {
    test_matrix_product<int>();
  }

  TEST(BLAS3_gemm, mdspan_double)
  {
    test_matrix_product<double>();
  }
}
