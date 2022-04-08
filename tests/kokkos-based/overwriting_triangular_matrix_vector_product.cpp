
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

struct MyExplicitDiagonalTag{};

template<class Triangle, class A_t, class x_t, class y_t>
void gold_solution_explicit_diagonal(Triangle /*t*/, A_t A, x_t x, y_t y)
{

  if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
  {
    for (std::size_t i=0; i<A.extent(0); ++i){
      y(i) = typename y_t::value_type{};
      for (std::size_t j=i; j<A.extent(1); ++j){
	y(i) += A(i,j) * x(j);
      }
    }
  }

  else{
    for (std::size_t i=0; i<A.extent(0); ++i){
      y(i) = typename y_t::value_type{};
      for (std::size_t j=0; j<=i; ++j){
	y(i) += A(i,j) * x(j);
      }
    }
  }
}

template<class Triangle, class A_t, class x_t, class y_t>
void gold_solution_implicit_unit_diagonal(Triangle /*t*/, A_t A, x_t x, y_t y)
{

  if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
  {
    for (std::size_t i=0; i<A.extent(0); ++i){
      y(i) = x(i);
      for (std::size_t j=i+1; j<A.extent(1); ++j){
	y(i) += A(i,j) * x(j);
      }
    }
  }

  else{
    for (std::size_t i=0; i<A.extent(0); ++i){
      y(i) = x(i);
      for (std::size_t j=0; j<i; ++j){
	y(i) += A(i,j) * x(j);
      }
    }
  }
}

template<bool explicitDiagonal, class A_t, class Triangle, class x_t, class y_t>
void kokkos_blas_overwriting_triangular_mat_vec_impl(A_t A, Triangle t, x_t x, y_t y)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  assert(A.extent(0) == A.extent(1));
  const std::size_t ext = A.extent(0);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute gold solution
  std::vector<value_type> gold(y.extent(0));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t y_gold(gold.data(), y.extent(0));

  if constexpr(explicitDiagonal){
    gold_solution_explicit_diagonal(t, A, x, y_gold);
    stdla::triangular_matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),
    					    A, t, MyExplicitDiagonalTag(), x, y);
  }
  else{
    gold_solution_implicit_unit_diagonal(t, A, x, y_gold);
    stdla::triangular_matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),
					    A, t, stdla::implicit_unit_diagonal_t(), x, y);
  }

  // after kernel, A,x should be unchanged, y should be equal to y_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_FLOAT_EQ(x(i), x_preKernel[i]);
      // for y, we need to use near or it won't work
      EXPECT_NEAR(y(i), y_gold(i), 1e-3);

      // remember that when using implicit unit diagonal, for testing purposes
      // the diagonal of A is filled with nans to actually check that the
      // implmentation does not access its diagonal as per spec,
      // so we cannot do a regular check to verify that diag(A) is same as
      // prekernel because a nan is not equal to itself, so need to treat it separately
      for (std::size_t j=0; j<ext; ++j){
	if (i!=j){
	  EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
	}
	else if (i==j && explicitDiagonal){
	  EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
	}
	else{
	  EXPECT_TRUE( std::isnan(A(i,j)) );
	  count++;
	}
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_preKernel[i]);
      // for y, we need to use near or it won't work
      EXPECT_NEAR(y(i), y_gold(i), 1e-12);

      // remember that when using implicit unit diagonal, for testing purposes
      // the diagonal of A is filled with nans to actually check that the
      // implmentation does not access its diagonal as per spec,
      // so we cannot do a regular check to verify that diag(A) is same as
      // prekernel because a nan is not equal to itself, so need to treat it separately
      for (std::size_t j=0; j<ext; ++j){
	if (i!=j){
	  EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
	}
	else if (i==j && explicitDiagonal){
	  EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
	}
	else{
	  EXPECT_TRUE( std::isnan(A(i,j)) );
	  count++;
	}
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>)
  {
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_DOUBLE_EQ(x(i).real(), x_preKernel[i].real());
      EXPECT_DOUBLE_EQ(x(i).imag(), x_preKernel[i].imag());
      EXPECT_NEAR(y(i).real(),	    y_gold(i).real(), 1e-12);
      EXPECT_NEAR(y(i).imag(),	    y_gold(i).imag(), 1e-12);

      // remember that when using implicit unit diagonal, for testing purposes
      // the diagonal of A is filled with nans to actually check that the
      // implmentation does not access its diagonal as per spec,
      // so we cannot do a regular check to verify that diag(A) is same as
      // prekernel because a nan is not equal to itself, so need to treat it separately
      for (std::size_t j=0; j<ext; ++j){
	if (i!=j){
	  EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	  EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
	}
	else if (i==j && explicitDiagonal){
	  EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	  EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
	}
	else{
	  EXPECT_TRUE( std::isnan(A(i,j).real()) );
	  EXPECT_TRUE( std::isnan(A(i,j).imag()) );
	  count++;
	}
      }
    }
  }

}
}//end anonym namespace

#define TEST_TRI_MAX_VEC_A(TRIANGLEPART)				\
  /* the fixture already properly fills A with random values */		\
  /* so we can use that directly without doing anything else */		\
  namespace stdla = std::experimental::linalg;				\
  constexpr bool explicitDiagonal = true;				\
  kokkos_blas_overwriting_triangular_mat_vec_impl<explicitDiagonal>(A_sym_e0, \
								    stdla::TRIANGLEPART, \
								    x_e0, y_e0); \

#define TEST_TRI_MAX_VEC_B(TRIANGLEPART)				\
  /* spec says that in such case, the diagonal of A is NOT accessed. */ \
  /* To test this, we fill the diagonal of A with nans such that */	\
  /* if diagonal is accessed, we get nans in y so test fails */		\
  									\
  for (std::size_t i=0; i<A_sym_e0.extent(0); ++i){			\
    A_sym_e0(i,i) = std::numeric_limits<value_type>::quiet_NaN();	\
  }									\
  namespace stdla = std::experimental::linalg;				\
  constexpr bool explicitDiagonal = false;				\
  kokkos_blas_overwriting_triangular_mat_vec_impl<explicitDiagonal>(A_sym_e0, \
								    stdla::TRIANGLEPART, \
								    x_e0, y_e0);

#define TEST_TRI_MAX_VEC_C(TRIANGLEPART)				\
  namespace stdla = std::experimental::linalg;				\
  using kc_t = Kokkos::complex<double>;					\
  using stdc_t = value_type;						\
  if (alignof(value_type) == alignof(kc_t)){				\
    constexpr bool explicitDiagonal = true;				\
    kokkos_blas_overwriting_triangular_mat_vec_impl<explicitDiagonal>(A_sym_e0, \
								      stdla::TRIANGLEPART, \
								      x_e0, y_e0); \
  }									\

#define TEST_TRI_MAX_VEC_D(TRIANGLEPART)				\
  namespace stdla = std::experimental::linalg;				\
  using kc_t = Kokkos::complex<double>;					\
  using stdc_t = value_type;						\
  if (alignof(value_type) == alignof(kc_t)){				\
    const auto nanVal = std::numeric_limits<double>::quiet_NaN();	\
    for (std::size_t i=0; i<A_sym_e0.extent(0); ++i){			\
      A_sym_e0(i,i) = {nanVal, nanVal};					\
    }									\
    constexpr bool explicitDiagonal = false;				\
    kokkos_blas_overwriting_triangular_mat_vec_impl<explicitDiagonal>(A_sym_e0, \
								      stdla::TRIANGLEPART, \
								      x_e0, y_e0); \
  }									\


#ifdef USE_UPPER
TEST_F(blas2_signed_float_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_A(upper_triangle);
}

TEST_F(blas2_signed_float_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_B(upper_triangle);
}

TEST_F(blas2_signed_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_A(upper_triangle);
}

TEST_F(blas2_signed_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_B(upper_triangle);
}

TEST_F(blas2_signed_complex_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_C(upper_triangle);
}

TEST_F(blas2_signed_complex_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_upper_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_D(upper_triangle);
}
#endif


#ifdef USE_LOWER
TEST_F(blas2_signed_float_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_A(lower_triangle);
}

TEST_F(blas2_signed_float_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_B(lower_triangle);
}

TEST_F(blas2_signed_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_A(lower_triangle);
}

TEST_F(blas2_signed_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_B(lower_triangle);
}

TEST_F(blas2_signed_complex_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_explicit_diagonal)
{
  TEST_TRI_MAX_VEC_C(lower_triangle);
}

TEST_F(blas2_signed_complex_double_fixture,
       kokkos_overwriting_triangular_matrix_vector_product_lower_implicit_unit_diagonal)
{
  TEST_TRI_MAX_VEC_D(lower_triangle);
}
#endif
