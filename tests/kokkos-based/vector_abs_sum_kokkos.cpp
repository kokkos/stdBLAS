
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace{

template<class x_t, class T>
auto vector_abs_sum_gold_solution(x_t x,
				  T initValue,
				  bool useInit)
{
  using std::abs;

  T result = {};
  for (std::size_t i=0; i<x.extent(0); ++i){
    result += abs(x(i));
  }

  if (useInit) result += initValue;
  return result;
}

template<class x_t, class T>
void kokkos_blas1_vector_abs_sum_test_impl(x_t x,
					   T initValue,
					   bool useInit)
{

  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x to verify it is not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);

  // compute gold
  const T gold = vector_abs_sum_gold_solution(x, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::vector_abs_sum(KokkosKernelsSTD::kokkos_exec<>(),
				   x, initValue);
  }else{
    result = stdla::vector_abs_sum(KokkosKernelsSTD::kokkos_exec<>(),
				   x);
  }

  if constexpr(std::is_same_v<value_type, float>){
    // cannot use EXPECT_FLOAT_EQ because
    // in some cases that fails on third digit or similr
    EXPECT_NEAR(result, gold, 1e-2);
  }

  if constexpr(std::is_same_v<value_type, double>){
    // similarly to float
    EXPECT_NEAR(result, gold, 1e-9);
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    EXPECT_NEAR(result, gold, 1e-9);
  }

  // x,y should not change after kernel
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i) == x_preKernel[i]);
  }

}
}//end anonym namespace


TEST_F(blas1_signed_float_fixture, kokkos_vector_abs_sum_noinitvalue)
{
  kokkos_blas1_vector_abs_sum_test_impl(x, static_cast<float>(0), false);
}

TEST_F(blas1_signed_float_fixture, kokkos_vector_abs_sum_initvalue)
{
  kokkos_blas1_vector_abs_sum_test_impl(x, static_cast<float>(3), true);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_abs_sum_noinitvalue)
{
  kokkos_blas1_vector_abs_sum_test_impl(x, static_cast<double>(0), false);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_abs_sum_initvalue)
{
  kokkos_blas1_vector_abs_sum_test_impl(x, static_cast<double>(5), true);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_abs_sum_noinitvalue)
{
  // for complex values, abs returns magnitude
  const double init = 0.;
  kokkos_blas1_vector_abs_sum_test_impl(x, init, false);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_abs_sum_initvalue)
{
  // for complex values, abs returns magnitude
  const double init = -2.;
  kokkos_blas1_vector_abs_sum_test_impl(x, init, true);
}
