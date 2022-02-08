
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace{

template<class A_t, class B_t>
void kokkos_blas_swap_test_rank2_impl(A_t A, B_t B)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // verify that A, B are different before running kernel
  for (std::size_t i=0; i<extent0; ++i){
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_TRUE(A(i,j)!=B(i,j));
    }
  }

  // copy A and B before kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto B_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(B);

  stdla::swap_elements(KokkosKernelsSTD::kokkos_exec<>(), A, B);

  // after kernel, A should be unchanged, B should be equal to A
  // note that we need to visit all elements rowwise since that is
  // how we stored above the preKernel values
  if constexpr(std::is_same_v<value_type, float>){
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_FLOAT_EQ(A(i,j), B_preKernel[count]);
	EXPECT_FLOAT_EQ(B(i,j), A_preKernel[count++]);
      }
    }
  }

  if constexpr(std::is_same_v<value_type, double>){
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j), B_preKernel[count]);
	EXPECT_DOUBLE_EQ(B(i,j), A_preKernel[count++]);
      }
    }
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), B_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), B_preKernel[count].imag());

	EXPECT_DOUBLE_EQ(B(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(B(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }
}
}//end anonym namespace

TEST_F(blas2_signed_float_fixture, kokkos_swap)
{
  kokkos_blas_swap_test_rank2_impl(A, B);
}

TEST_F(blas2_signed_double_fixture, kokkos_swap)
{
  kokkos_blas_swap_test_rank2_impl(A, B);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_swap)
{
  using kc_t   = Kokkos::complex<double>;
  using stdc_t = value_type;
  if (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_swap_test_rank2_impl(A, B);
  }
}
