#include "./gtest_fixtures.hpp"
#include <iostream>

namespace {
  using LinearAlgebra::triangular_matrix_vector_product;
  using LinearAlgebra::transposed;
  using LinearAlgebra::explicit_diagonal_t;
  using LinearAlgebra::implicit_unit_diagonal_t;
  using LinearAlgebra::lower_triangle_t;
  using LinearAlgebra::upper_triangle_t;
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

  template<class Scalar, class Triangle, class Diagonal, int NumMdspanArgs>
  void test_matrix_product()
  {
    using scalar_t = Scalar;
    using real_t = typename Magnitude<Scalar>::type;

    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using matrix_t = mdspan<scalar_t, extents_t, layout_left>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>, layout_left>;

    constexpr std::size_t maxDim = 7;
    constexpr std::size_t storageSize(maxDim*maxDim + 4*maxDim);
    std::vector<scalar_t> storage(storageSize);

    for (std::size_t numRows : {1, 4, 7}) {
      std::size_t numCols = numRows;
      std::size_t offset = 0;
      matrix_t A(storage.data() + offset, numRows, numCols);
      offset += numRows * numCols;
      vector_t x(storage.data() + offset, numCols);
      offset += numCols;
      vector_t y(storage.data() + offset, numRows);
      offset += numRows;
      vector_t z(storage.data() + offset, numRows);
      offset += numRows;
      vector_t gs(storage.data() + offset, numRows);

      fill_matrix(A, scalar_t(real_t(1)));
      fill_vector(x, scalar_t(real_t(2)));
      fill_vector(z, scalar_t(real_t(2)));

      // Initialize vector to zero
      for (std::size_t i = 0; i < numRows; i++) {
        if(NumMdspanArgs <= 3)
        {
          gs(i) = scalar_t(0.0);
        }
        else
        {
          gs(i) = z(i);
        }
      }

      // Perform matvec
      for (std::size_t j = 0; j < numCols; ++j) {
        if(std::is_same_v<Triangle, lower_triangle_t>)
        {
            for (std::size_t i = j+1; i < numRows; ++i) {
              gs(i) += A(i,j) * x(j);
            }
        }
        else
        {
            for (std::size_t i = 0; i < j; ++i) {
              gs(i) += A(i,j) * x(j);
            }
        }
        if(std::is_same_v<Diagonal, explicit_diagonal_t>)
        {
              gs(j) += A(j,j) * x(j);
        }
        else
        {
              gs(j) += x(j);
        }
      }

      // Fill result vector with flag values to make sure that we
      // computed everything.
      for (std::size_t j = 0; j < numRows; ++j) {
        y(j) = std::numeric_limits<scalar_t>::min();
      }

      if(NumMdspanArgs == 2)
      {
        // In-place version
        triangular_matrix_vector_product(A, Triangle{}, Diagonal{}, x);

        for (std::size_t i = 0; i < numRows; ++i) {
          EXPECT_DOUBLE_EQ(x(i), gs(i)) << "Vectors differ at index " << i;
        }
      }
      else if(NumMdspanArgs == 3)
      {
        // Overwriting version
        triangular_matrix_vector_product(A, Triangle{}, Diagonal{}, x, y);

        for (std::size_t i = 0; i < numRows; ++i) {
          EXPECT_DOUBLE_EQ(y(i), gs(i)) << "Vectors differ at index " << i;
        }
      }
      else if(NumMdspanArgs == 4)
      {
        // Updating version
        triangular_matrix_vector_product(A, Triangle{}, Diagonal{}, x, z, y);

        for (std::size_t i = 0; i < numRows; ++i) {
          EXPECT_DOUBLE_EQ(y(i), gs(i)) << "Vectors differ at index " << i;
        }
      }

    } // numRows
  }

  TEST(BLAS2_trmv_u_e_2args, mdspan_double)
  {
    test_matrix_product<double, upper_triangle_t, explicit_diagonal_t, 2>();
  }
  TEST(BLAS2_trmv_l_e_2args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, explicit_diagonal_t, 2>();
  }

  TEST(BLAS2_trmv_u_i_2args, mdspan_double)
  {
        test_matrix_product<double, upper_triangle_t, implicit_unit_diagonal_t, 2>();
  }
  TEST(BLAS2_trmv_l_i_2args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, implicit_unit_diagonal_t, 2>();
  }

  TEST(BLAS2_trmv_u_e_3args, mdspan_double)
  {
    test_matrix_product<double, upper_triangle_t, explicit_diagonal_t, 3>();
  }
  TEST(BLAS2_trmv_l_e_3args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, explicit_diagonal_t, 3>();
  }

  TEST(BLAS2_trmv_u_i_3args, mdspan_double)
  {
        test_matrix_product<double, upper_triangle_t, implicit_unit_diagonal_t, 3>();
  }
  TEST(BLAS2_trmv_l_i_3args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, implicit_unit_diagonal_t, 3>();
  }

  TEST(BLAS2_trmv_u_e_4args, mdspan_double)
  {
    test_matrix_product<double, upper_triangle_t, explicit_diagonal_t, 4>();
  }
  TEST(BLAS2_trmv_l_e_4args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, explicit_diagonal_t, 4>();
  }

  TEST(BLAS2_trmv_u_i_4args, mdspan_double)
  {
        test_matrix_product<double, upper_triangle_t, implicit_unit_diagonal_t, 4>();
  }
  TEST(BLAS2_trmv_l_i_4args, mdspan_double)
  {
    test_matrix_product<double, lower_triangle_t, implicit_unit_diagonal_t, 4>();
  }
}
