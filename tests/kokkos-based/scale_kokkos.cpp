
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class x_t, class FactorT>
void scale_gold_solution(x_t x, FactorT factor)
{
  FactorT result = {};
  for (std::size_t i=0; i<x.extent(0); ++i){
    x(i) *= factor;
  }
}

template<class x_t, class FactorT>
void kokkos_blas1_scale_test_impl(x_t x, FactorT factor)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // compute gold
  std::vector<value_type> gold(extent);
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t x_gold(gold.data(), extent);
  for (std::size_t i=0; i<x.extent(0); ++i){
    x_gold(i) = x(i);
  }
  scale_gold_solution(x_gold, factor);

  stdla::scale(KokkosKernelsSTD::kokkos_exec<>(), factor, x);

  if constexpr(std::is_same_v<value_type, float>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_FLOAT_EQ(x(i), x_gold(i));
    }
  }

  if constexpr(std::is_same_v<value_type, double>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_gold(i));
    }
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i).real(), x_gold(i).real());
      EXPECT_DOUBLE_EQ(x(i).imag(), x_gold(i).imag());
    }
  }

}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, kokkos_scale)
{
  kokkos_blas1_scale_test_impl(x, static_cast<value_type>(2));
}

TEST_F(blas1_signed_double_fixture, kokkos_scale)
{
  kokkos_blas1_scale_test_impl(x, static_cast<value_type>(2));
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_scale_a)
{
  const value_type factor{2., 0.};
  kokkos_blas1_scale_test_impl(x, factor);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_scale_b)
{
  kokkos_blas1_scale_test_impl(x, 2.);
}
