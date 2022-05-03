
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t, class B_t, class C_t>
void gemm_gold_solution(A_t A, B_t B, C_t C)
{
  for (std::size_t i=0; i<C.extent(0); ++i){
    for (std::size_t j=0; j<C.extent(1); ++j){
      C(i,j) = typename C_t::value_type{};
      for (std::size_t k=0; k<B.extent(0); ++k){
	C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

template<class A_t, class B_t, class C_t>
void kokkos_blas_gemm_impl(A_t A, B_t B, C_t C)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);
  const std::size_t extent2 = B.extent(1);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto B_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(B);
  auto C_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(C);

  // compute gold gemm
  std::vector<value_type> gold(extent0*extent2);
  using mdspan_t = mdspan<value_type, extents<dynamic_extent, dynamic_extent>>;
  mdspan_t C_gold(gold.data(), extent0, extent2);
  gemm_gold_solution(A, B, C_gold);

  stdla::matrix_product(KokkosKernelsSTD::kokkos_exec<>(), A, B, C);

  // after kernel, A,B should be unchanged, C should be equal to C_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<A.extent(0); ++i){
      for (std::size_t j=0; j<A.extent(1); ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<B.extent(0); ++i){
      for (std::size_t j=0; j<B.extent(1); ++j){
	EXPECT_FLOAT_EQ(B(i,j), B_preKernel[count++]);
      }
    }

    // check C
    for (std::size_t i=0; i<C.extent(0); ++i){
      for (std::size_t j=0; j<C.extent(1); ++j){
	EXPECT_NEAR(C(i,j), C_gold(i,j), 1e-3);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<A.extent(0); ++i){
      for (std::size_t j=0; j<A.extent(1); ++j){
	EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<B.extent(0); ++i){
      for (std::size_t j=0; j<B.extent(1); ++j){
	EXPECT_DOUBLE_EQ(B(i,j), B_preKernel[count++]);
      }
    }

    // check C
    for (std::size_t i=0; i<C.extent(0); ++i){
      for (std::size_t j=0; j<C.extent(1); ++j){
	EXPECT_NEAR(C(i,j), C_gold(i,j), 1e-9);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<A.extent(0); ++i){
      for (std::size_t j=0; j<A.extent(1); ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<B.extent(0); ++i){
      for (std::size_t j=0; j<B.extent(1); ++j){
	EXPECT_DOUBLE_EQ(B(i,j).real(), B_preKernel[count].real());
	EXPECT_DOUBLE_EQ(B(i,j).imag(), B_preKernel[count++].imag());
      }
    }

    // check C
    for (std::size_t i=0; i<C.extent(0); ++i){
      for (std::size_t j=0; j<C.extent(1); ++j){
	EXPECT_NEAR(C(i,j).real(), C_gold(i,j).real(), 1e-9);
	EXPECT_NEAR(C(i,j).imag(), C_gold(i,j).imag(), 1e-9);
      }
    }
  }
}
}//end anonym namespace

TEST_F(blas3_signed_float_fixture, kokkos_gemm_C_AB)
{
  kokkos_blas_gemm_impl(A_e0e1, B_e1e2, C_e0e2);
}

TEST_F(blas3_signed_double_fixture, kokkos_gemm_C_AB)
{
  kokkos_blas_gemm_impl(A_e0e1, B_e1e2, C_e0e2);
}

TEST_F(blas3_signed_complex_double_fixture, kokkos_gemm_C_AB)
{
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_gemm_impl(A_e0e1, B_e1e2, C_e0e2);
  }
}
