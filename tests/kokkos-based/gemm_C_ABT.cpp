
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

  auto BT = stdla::transposed(B);

  // copy operands before running the kernel
  auto A_preKernel  = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto BT_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(BT);
  auto C_preKernel  = kokkostesting::create_stdvector_and_copy_rowwise(C);

  // compute gold gemm
  std::vector<value_type> gold(A.extent(0)*BT.extent(1));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent, dynamic_extent>>;
  mdspan_t C_gold(gold.data(), A.extent(0), BT.extent(1));
  gemm_gold_solution(A, BT, C_gold);

  stdla::matrix_product(KokkosKernelsSTD::kokkos_exec<>(), A, BT, C);

  // after kernel, A,BT should be unchanged, C should be equal to C_gold.
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

    // check BT
    count=0;
    for (std::size_t i=0; i<BT.extent(0); ++i){
      for (std::size_t j=0; j<BT.extent(1); ++j){
	EXPECT_FLOAT_EQ(BT(i,j), BT_preKernel[count++]);
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

    // check BT
    count=0;
    for (std::size_t i=0; i<BT.extent(0); ++i){
      for (std::size_t j=0; j<BT.extent(1); ++j){
	EXPECT_DOUBLE_EQ(BT(i,j), BT_preKernel[count++]);
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

    // check BT
    count=0;
    for (std::size_t i=0; i<BT.extent(0); ++i){
      for (std::size_t j=0; j<BT.extent(1); ++j){
	EXPECT_DOUBLE_EQ(BT(i,j).real(), BT_preKernel[count].real());
	EXPECT_DOUBLE_EQ(BT(i,j).imag(), BT_preKernel[count++].imag());
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

TEST_F(blas3_signed_float_fixture, kokkos_gemm_C_ATB)
{
  kokkos_blas_gemm_impl(A_e0e1, B_e2e1, C_e0e2);
}

TEST_F(blas3_signed_double_fixture, kokkos_gemm_C_ATB)
{
  kokkos_blas_gemm_impl(A_e0e1, B_e2e1, C_e0e2);
}

TEST_F(blas3_signed_complex_double_fixture, kokkos_gemm_C_ATB)
{
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_gemm_impl(A_e0e1, B_e2e1, C_e0e2);
  }
}
