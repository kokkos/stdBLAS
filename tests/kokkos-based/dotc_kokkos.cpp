
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class x_t, class y_t, class T>
auto dotc_gold_solution(x_t x, y_t y, T initValue, bool useInit)
{

  T result = {};
  if (useInit) result = initValue;

  for (std::size_t i=0; i<x.extent(0); ++i){
    result += std::conj(x(i)) * y(i);
  }

  return result;
}

template<class x_t, class y_t, class T>
void kokkos_blas1_dotc_test_impl(x_t x, y_t y, T initValue, bool useInit)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x and y to verify they are not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute gold
  const auto gold = dotc_gold_solution(x, y, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::dotc(KokkosKernelsSTD::kokkos_exec<>(), x, y, initValue);
  }else{
    result = stdla::dotc(KokkosKernelsSTD::kokkos_exec<>(), x, y);
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>)
  {
    EXPECT_NEAR(result.real(), gold.real(), 1e-9);
    EXPECT_NEAR(result.imag(), gold.imag(), 1e-9);

    for (std::size_t i=0; i<extent; ++i){
      EXPECT_TRUE(x(i) == x_preKernel[i]);
      EXPECT_TRUE(y(i) == y_preKernel[i]);
    }
  }
}
}//end anonym namespace

TEST_F(blas1_signed_complex_double_fixture, kokkos_dotc_noinitvalue)
{
  using kc_t   = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    const value_type init{0., 0.};
    kokkos_blas1_dotc_test_impl(x, y, init, false);
  }
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_dotc_initvalue)
{
  using kc_t   = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    const value_type init{-4., 5.};
    kokkos_blas1_dotc_test_impl(x, y, init, true);
  }
}
