
#include "gtest_fixtures.hpp"
#include "helpers.hpp"
#include <cmath>

namespace
{

template<class x_t, class T>
T vector_norm2_gold_solution(x_t x, T initValue, bool useInit)
{
  using std::abs;
  using value_type = typename x_t::value_type;

  T result = {};
  for (std::size_t i=0; i<x.extent(0); ++i){
    if constexpr(std::is_same_v<value_type, std::complex<double>>){
      result += std::norm(x(i));
    }
    else{
      result += x(i) * x(i);
    }
  }

  using std::sqrt;
  if (useInit){
    return sqrt(initValue + result);
  }
  else{
    return sqrt(result);
  }
}

template<class x_t, class T>
void kokkos_blas1_vector_norm2_test_impl(x_t x, T initValue, bool useInit)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x to verify they are not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);

  const T gold = vector_norm2_gold_solution(x, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::vector_norm2(KokkosKernelsSTD::kokkos_exec<>(),
				 x, initValue);
  }else{
    result = stdla::vector_norm2(KokkosKernelsSTD::kokkos_exec<>(),
				 x);
  }

  if constexpr(std::is_same_v<value_type, float>){
    EXPECT_NEAR(result, gold, 1e-2);
  }

  if constexpr(std::is_same_v<value_type, double>){
    EXPECT_NEAR(result, gold, 1e-9);
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    EXPECT_NEAR(result, gold, 1e-9);
  }

  // x should not change after kernel
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i) == x_preKernel[i]);
  }

}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, kokkos_vector_norm2_noinitvalue)
{
  kokkos_blas1_vector_norm2_test_impl(x, static_cast<float>(0), false);
}

TEST_F(blas1_signed_float_fixture, kokkos_vector_norm2_initvalue)
{
  kokkos_blas1_vector_norm2_test_impl(x, static_cast<float>(3), true);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_norm2_noinitvalue)
{
  kokkos_blas1_vector_norm2_test_impl(x, static_cast<double>(0), false);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_norm2_initvalue)
{
  kokkos_blas1_vector_norm2_test_impl(x, static_cast<double>(5), true);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_norm2_noinitvalue)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas1_vector_norm2_test_impl(x, static_cast<double>(0), false);
  }
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_norm2_initvalue)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_blas1_vector_norm2_test_impl(x, static_cast<double>(5), true);
  }
}
