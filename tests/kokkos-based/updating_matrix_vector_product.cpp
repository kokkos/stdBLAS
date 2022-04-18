
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t, class x_t, class y_t, class z_t>
void gemv_gold_solution(A_t A, x_t x, y_t y, z_t z)
{
  for (std::size_t i=0; i<A.extent(0); ++i){
    z(i) = typename z_t::value_type{};
    for (std::size_t j=0; j<A.extent(1); ++j){
      z(i) += y(i) + A(i,j) * x(j);
    }
  }
}

template<class A_t, class x_t, class y_t, class z_t>
void kokkos_blas_updating_gemv_impl(A_t A, x_t x, y_t y, z_t z)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);
  auto z_preKernel = kokkostesting::create_stdvector_and_copy(z);

  // compute gold gemv
  std::vector<value_type> gold(z.extent(0));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t z_gold(gold.data(), z.extent(0));
  gemv_gold_solution(A, x, y, z_gold);

  stdla::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(), A, x, y, z);

  // after kernel, A,x,y should be unchanged, z should be equal to z_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_FLOAT_EQ(x(j), x_preKernel[j]);
    }

    // check A, z, y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_FLOAT_EQ(z(i), z_gold(i));
      EXPECT_FLOAT_EQ(y(i), y_preKernel[i]);
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_DOUBLE_EQ(x(j), x_preKernel[j]);
    }

    // check A, y, z
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_FLOAT_EQ(z(i), z_gold(i));
      EXPECT_FLOAT_EQ(y(i), y_preKernel[i]);
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_DOUBLE_EQ(x(j).real(), x_preKernel[j].real());
      EXPECT_DOUBLE_EQ(x(j).imag(), x_preKernel[j].imag());
    }

    // check A, y, z
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_DOUBLE_EQ(z(i).real(), z_gold(i).real());
      EXPECT_DOUBLE_EQ(z(i).imag(), z_gold(i).imag());

      EXPECT_DOUBLE_EQ(y(i).real(), y_preKernel[i].real());
      EXPECT_DOUBLE_EQ(y(i).imag(), y_preKernel[i].imag());

      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }

}
}//end anonym namespace

TEST_F(blas2_signed_float_fixture, kokkos_updating_matrix_vector_product)
{
  kokkos_blas_updating_gemv_impl(A_e0e1, x_e1, x_e0, y_e0);
}

TEST_F(blas2_signed_double_fixture, kokkos_updating_matrix_vector_product)
{
  kokkos_blas_updating_gemv_impl(A_e0e1, x_e1, x_e0, y_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_updating_matrix_vector_product)
{
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_updating_gemv_impl(A_e0e1, x_e1, x_e0, y_e0);
  }
}
