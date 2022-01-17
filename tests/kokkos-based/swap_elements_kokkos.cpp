
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"

namespace{

template<class x_t, class y_t>
void kokkos_blas1_swap_test_impl(x_t x, y_t y)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // verify that x, y are different before running kernel
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i)!=y(i));
  }

  // copy x and y before kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  stdla::swap_elements(KokkosKernelsSTD::kokkos_exec<>(), x, y);

  // after kernel, x should be unchanged, y should be equal to x
  if constexpr(std::is_same_v<value_type, float>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_FLOAT_EQ(x(i), y_preKernel[i]);
      EXPECT_FLOAT_EQ(y(i), x_preKernel[i]);
    }
  }

  if constexpr(std::is_same_v<value_type, double>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i), y_preKernel[i]);
      EXPECT_DOUBLE_EQ(y(i), x_preKernel[i]);
    }
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i).real(), y_preKernel[i].real());
      EXPECT_DOUBLE_EQ(x(i).imag(), y_preKernel[i].imag());

      EXPECT_DOUBLE_EQ(y(i).real(), x_preKernel[i].real());
      EXPECT_DOUBLE_EQ(y(i).imag(), x_preKernel[i].imag());
    }
  }

}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, kokkos_swap)
{
  kokkos_blas1_swap_test_impl(x, y);
}

TEST_F(blas1_signed_double_fixture, kokkos_swap)
{
  kokkos_blas1_swap_test_impl(x, y);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_swap)
{
  kokkos_blas1_swap_test_impl(x, y);
}
