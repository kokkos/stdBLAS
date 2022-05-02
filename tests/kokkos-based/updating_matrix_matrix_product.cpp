
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t, class B_t, class E_t, class C_t>
void gemm_gold_solution(A_t A, B_t B, E_t E, C_t C)
{
  for (std::size_t i=0; i<C.extent(0); ++i){
    for (std::size_t j=0; j<C.extent(1); ++j){
      C(i,j) = E(i,j);
      for (std::size_t k=0; k<B.extent(0); ++k){
	C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

template<class A_t, class B_t, class E_t, class C_t>
void kokkos_blas_updating_gemm_impl(A_t A, B_t B, E_t E, C_t C)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);
  const std::size_t extent2 = B.extent(1);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto B_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(B);
  auto E_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(E);
  auto C_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(C);

  // compute gold gemm
  std::vector<value_type> gold(extent0*extent2);
  using mdspan_t = mdspan<value_type, extents<dynamic_extent, dynamic_extent>>;
  mdspan_t C_gold(gold.data(), extent0, extent2);
  gemm_gold_solution(A, B, E, C_gold);

  stdla::matrix_product(KokkosKernelsSTD::kokkos_exec<>(), A, B, E, C);

  // after kernel, A,B should be unchanged, C should be equal to C_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<extent1; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_FLOAT_EQ(B(i,j), B_preKernel[count++]);
      }
    }

    // check C, E
    count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_FLOAT_EQ(E(i,j), E_preKernel[count++]);
	EXPECT_NEAR(C(i,j), C_gold(i,j), 1e-3);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<extent1; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_DOUBLE_EQ(B(i,j), B_preKernel[count++]);
      }
    }

    // check C, E
    count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_DOUBLE_EQ(E(i,j), E_preKernel[count++]);
	EXPECT_NEAR(C(i,j), C_gold(i,j), 1e-9);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>){
    // check A
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }

    // check B
    count=0;
    for (std::size_t i=0; i<extent1; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_DOUBLE_EQ(B(i,j).real(), B_preKernel[count].real());
	EXPECT_DOUBLE_EQ(B(i,j).imag(), B_preKernel[count++].imag());
      }
    }

    // check C, E
    count=0;
    for (std::size_t i=0; i<extent0; ++i){
      for (std::size_t j=0; j<extent2; ++j){
	EXPECT_DOUBLE_EQ(E(i,j).real(), E_preKernel[count].real());
	EXPECT_DOUBLE_EQ(E(i,j).imag(), E_preKernel[count++].imag());

	EXPECT_NEAR(C(i,j).real(), C_gold(i,j).real(), 1e-9);
	EXPECT_NEAR(C(i,j).imag(), C_gold(i,j).imag(), 1e-9);
      }
    }
  }
}
}//end anonym namespace

TEST_F(blas2_signed_float_fixture, kokkos_updating_matrix_matrix_product)
{
  kokkos_blas_updating_gemm_impl(A_e0e1, B_e1e2, E_e0e2, C_e0e2);
}

// TEST_F(blas2_signed_double_fixture, kokkos_updating_matrix_vector_product)
// {
//   kokkos_blas_updating_gemm_impl(A_e0e1, B_e1e2, C_e0e2);
// }

// TEST_F(blas2_signed_complex_double_fixture, kokkos_updating_matrix_vector_product)
// {
//   using kc_t = Kokkos::complex<double>;
//   using stdc_t = value_type;
//   if constexpr (alignof(value_type) == alignof(kc_t)){
//     kokkos_blas_updating_gemm_impl(A_e0e1, B_e1e2, C_e0e2);
//   }
// }
