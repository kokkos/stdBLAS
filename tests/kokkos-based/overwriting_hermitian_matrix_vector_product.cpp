
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t>
void verify_hermitian(A_t A)
{
  using value_type = typename A_t::value_type;

  if constexpr(std::is_same_v<value_type, std::complex<double>>)
  {
    for (std::size_t i=0; i<A.extent(0); ++i)
    {
      EXPECT_DOUBLE_EQ( A(i,i).imag(), static_cast<double>(0));
      for (std::size_t j=i+1; j<A.extent(1); ++j){
	EXPECT_TRUE( A(i,j) == std::conj(A(j,i)) );
      }
    }
  }

  else{
    for (std::size_t i=0; i<A.extent(0); ++i)
    {
      for (std::size_t j=i; j<A.extent(1); ++j){
	EXPECT_TRUE( A(i,j) == A(j,i) );
      }
    }
  }
}

template<class A_t, class x_t, class y_t>
void gemv_gold_solution(A_t A, x_t x, y_t y)
{
  for (std::size_t i=0; i<A.extent(0); ++i){
    y(i) = typename y_t::value_type{};
    for (std::size_t j=0; j<A.extent(1); ++j){
      y(i) += A(i,j) * x(j);
    }
  }
}

template<class A_t, class Triangle, class x_t, class y_t>
void kokkos_blas_overwriting_hemv_impl(A_t A, Triangle t, x_t x, y_t y)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  assert(A.extent(0) == A.extent(1));
  const std::size_t ext = A.extent(0);

  verify_hermitian(A);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute y gold hemv
  std::vector<value_type> gold(y.extent(0));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t y_gold(gold.data(), y.extent(0));
  gemv_gold_solution(A, x, y_gold);

  stdla::hermitian_matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),
					 A, t, x, y);

  // after kernel, A,x should be unchanged, y should be equal to y_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_FLOAT_EQ(x(i), x_preKernel[i]);
      // for y, we need to use near or it won't work
      EXPECT_NEAR(y(i), y_gold(i), 1e-3);
      for (std::size_t j=0; j<ext; ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_preKernel[i]);
      // for y, we need to use near or it won't work
      EXPECT_NEAR(y(i), y_gold(i), 1e-12);
      for (std::size_t j=0; j<ext; ++j){
	EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i){
      EXPECT_DOUBLE_EQ(x(i).real(), x_preKernel[i].real());
      EXPECT_DOUBLE_EQ(x(i).imag(), x_preKernel[i].imag());
      EXPECT_NEAR(y(i).real(),	    y_gold(i).real(), 1e-12);
      EXPECT_NEAR(y(i).imag(),	    y_gold(i).imag(), 1e-12);

      for (std::size_t j=0; j<ext; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }

}
}//end anonym namespace

#ifdef USE_UPPER
TEST_F(blas2_signed_float_fixture, kokkos_overwriting_hem_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_overwriting_hemv_impl(A_sym_e0, stdla::upper_triangle, x_e0, y_e0);
}

TEST_F(blas2_signed_double_fixture, kokkos_overwriting_hem_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_overwriting_hemv_impl(A_sym_e0, stdla::upper_triangle, x_e0, y_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_overwriting_hem_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_overwriting_hemv_impl(A_hem_e0, stdla::upper_triangle, x_e0, y_e0);
  }
}
#endif

#ifdef USE_LOWER
TEST_F(blas2_signed_float_fixture, kokkos_overwriting_hem_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_overwriting_hemv_impl(A_sym_e0, stdla::lower_triangle, x_e0, y_e0);
}

TEST_F(blas2_signed_double_fixture, kokkos_overwriting_hem_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_overwriting_hemv_impl(A_sym_e0, stdla::lower_triangle, x_e0, y_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_overwriting_hem_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_overwriting_hemv_impl(A_hem_e0, stdla::lower_triangle, x_e0, y_e0);
  }
}
#endif
