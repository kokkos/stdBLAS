
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t, class x_t, class y_t, class z_t>
void gemv_gold_solution(A_t A, x_t x, y_t y, z_t z)
{
  for (std::size_t i=0; i<A.extent(0); ++i){
    typename y_t::value_type sum{};
    for (std::size_t j=0; j<A.extent(1); ++j){
      sum += A(i,j) * x(j);
    }
    z(i) = y(i) + sum;
  }
}

template<class A_t, class Triangle, class x_t, class y_t, class z_t>
void kokkos_blas_updating_symv_impl(A_t A, Triangle t, x_t x, y_t y, z_t z)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  assert(A.extent(0) == A.extent(1));
  const std::size_t ext = A.extent(0);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);
  auto z_preKernel = kokkostesting::create_stdvector_and_copy(z);

  // compute gold solution
  std::vector<value_type> gold(y.extent(0));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t z_gold(gold.data(), z.extent(0));
  gemv_gold_solution(A, x, y, z_gold);

  stdla::symmetric_matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),
					 A, t, x, y, z);

  // after kernel, A,x,y should be unchanged, z should be equal to z_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i)
    {
      EXPECT_FLOAT_EQ(x(i), x_preKernel[i]);
      EXPECT_FLOAT_EQ(y(i), y_preKernel[i]);
      // use near or it won't work
      EXPECT_NEAR(z(i), z_gold(i), 1e-3);

      for (std::size_t j=0; j<ext; ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    std::size_t count=0;
    for (std::size_t i=0; i<ext; ++i)
    {
      EXPECT_DOUBLE_EQ(x(i), x_preKernel[i]);
      EXPECT_DOUBLE_EQ(y(i), y_preKernel[i]);
      // use near or it won't work
      EXPECT_NEAR(z(i), z_gold(i), 1e-12);

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
      EXPECT_DOUBLE_EQ(y(i).real(), y_preKernel[i].real());
      EXPECT_DOUBLE_EQ(y(i).imag(), y_preKernel[i].imag());
      EXPECT_NEAR(z(i).real(),      z_gold(i).real(), 1e-12);
      EXPECT_NEAR(z(i).imag(),      z_gold(i).imag(), 1e-12);

      for (std::size_t j=0; j<ext; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }

}
}//end anonym namespace

#ifdef USE_UPPER
TEST_F(blas2_signed_float_fixture, kokkos_updating_sym_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_updating_symv_impl(A_sym_e0, stdla::upper_triangle, x_e0, y_e0, z_e0);
}

TEST_F(blas2_signed_double_fixture, kokkos_updating_sym_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_updating_symv_impl(A_sym_e0, stdla::upper_triangle, x_e0, y_e0, z_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_updating_sym_matrix_vector_product_upper)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_updating_symv_impl(A_sym_e0, stdla::upper_triangle, x_e0, y_e0, z_e0);
  }
}
#endif

#ifdef USE_LOWER
TEST_F(blas2_signed_float_fixture, kokkos_updating_sym_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_updating_symv_impl(A_sym_e0, stdla::lower_triangle, x_e0, y_e0, z_e0);
}

TEST_F(blas2_signed_double_fixture, kokkos_updating_sym_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  kokkos_blas_updating_symv_impl(A_sym_e0, stdla::lower_triangle, x_e0, y_e0, z_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_updating_sym_matrix_vector_product_lower)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_updating_symv_impl(A_sym_e0, stdla::lower_triangle, x_e0, y_e0, z_e0);
  }
}
#endif
