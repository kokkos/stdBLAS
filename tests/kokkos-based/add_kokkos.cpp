
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"

namespace{

template<class x_t, class y_t, class z_t>
void add_gold_solution(x_t x, y_t y, z_t z)
{
  for (std::size_t i=0; i<x.extent(0); ++i){
    z(i) = x(i) + y(i);
  }
}

template<class x_t, class y_t, class z_t>
void kokkos_blas1_add_test_impl(x_t x, y_t y, z_t z)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x and y to verify they are not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute gold
  std::vector<value_type> gold(extent);
  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t z_gold(gold.data(), extent);
  add_gold_solution(x, y, z_gold);

  stdla::add(KokkosKernelsSTD::kokkos_exec<>(), x, y, z);

  if constexpr(std::is_same_v<value_type, float>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_FLOAT_EQ(x(i), x_preKernel[i]);
      EXPECT_FLOAT_EQ(y(i), y_preKernel[i]);
      EXPECT_FLOAT_EQ(z(i), z_gold(i));
    }
  }

  if constexpr(std::is_same_v<value_type, double>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_preKernel[i]);
      EXPECT_DOUBLE_EQ(y(i), y_preKernel[i]);
      EXPECT_DOUBLE_EQ(z(i), z_gold(i));
    }
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_TRUE(x(i) == x_preKernel[i]);
      EXPECT_TRUE(y(i) == y_preKernel[i]);
      EXPECT_DOUBLE_EQ(z(i).real(), z_gold[i].real());
      EXPECT_DOUBLE_EQ(z(i).imag(), z_gold[i].imag());
    }
  }

}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, kokkos_add)
{
  kokkos_blas1_add_test_impl(x, y, z);
}

TEST_F(blas1_signed_double_fixture, kokkos_add)
{
  kokkos_blas1_add_test_impl(x, y, z);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_add)
{
  kokkos_blas1_add_test_impl(x, y, z);
}
