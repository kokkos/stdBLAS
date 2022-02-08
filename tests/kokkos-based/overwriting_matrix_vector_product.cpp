
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

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

template<class A_t, class x_t, class y_t>
void kokkos_blas_overwriting_gemv_impl(A_t A, x_t x, y_t y)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // copy operands before running the kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute y gold gemv
  std::vector<value_type> gold(y.extent(0));
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t y_gold(gold.data(), y.extent(0));
  gemv_gold_solution(A, x, y_gold);

  stdla::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(), A, x, y);

  // after kernel, A,x should be unchanged, y should be equal to y_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_FLOAT_EQ(x(j), x_preKernel[j]);
    }

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_FLOAT_EQ(y(i), y_gold(i));
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

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_DOUBLE_EQ(y(i), y_gold(i));
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

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_DOUBLE_EQ(y(i).real(), y_gold(i).real());
      EXPECT_DOUBLE_EQ(y(i).imag(), y_gold(i).imag());

      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }

}
}//end anonym namespace

TEST_F(blas2_signed_float_fixture, kokkos_overwriting_matrix_vector_product)
{
  kokkos_blas_overwriting_gemv_impl(A, v_Ae1, v_Ae0);
}

TEST_F(blas2_signed_double_fixture, kokkos_overwriting_matrix_vector_product)
{
  kokkos_blas_overwriting_gemv_impl(A, v_Ae1, v_Ae0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_overwriting_matrix_vector_product)
{
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if (alignof(value_type) == alignof(kc_t)){
    kokkos_blas_overwriting_gemv_impl(A, v_Ae1, v_Ae0);
  }
}
