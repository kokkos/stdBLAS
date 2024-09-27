
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class x_t>
std::experimental::extents<>::size_type
vector_idx_abs_max_gold_solution(x_t x)
{

  using std::abs;
  using size_type = std::experimental::extents<>::size_type;

  size_type maxInd = 0;
  decltype(abs(x(0))) maxVal = abs(x(0));
  for (size_type i = 1; i < x.extent(0); ++i) {
    if (maxVal < abs(x(i))) {
      maxVal = abs(x(i));
      maxInd = i;
    }
  }

  return maxInd;
}

template<class x_t>
void kokkos_blas1_vector_idx_abs_max_test_impl(x_t x)
{

  namespace stdla = std::experimental::linalg;

  // copy x to verify it is not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);

  const auto gold = vector_idx_abs_max_gold_solution(x);
  const auto result = stdla::vector_idx_abs_max(KokkosKernelsSTD::kokkos_exec<>(), x);
  EXPECT_TRUE(gold == result);
  static_assert(std::is_same_v<decltype(gold), decltype(result)>,
		"test:vector_idx_abs_max: gold and result types not same");

  // x should not change after kernel
  const std::size_t extent = x.extent(0);
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i) == x_preKernel[i]);
  }

}
}//end anonym namespace


TEST_F(blas1_signed_float_fixture, kokkos_vector_idx_abs_max)
{
  kokkos_blas1_vector_idx_abs_max_test_impl(x);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_idx_abs_max)
{
  kokkos_blas1_vector_idx_abs_max_test_impl(x);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_idx_abs_max)
{
  using kc_t   = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr(alignof(value_type) == alignof(kc_t)){
    kokkos_blas1_vector_idx_abs_max_test_impl(x);
  }
}
