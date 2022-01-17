
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"
#include <cmath>

namespace
{

template<class x_t, class T>
std::experimental::linalg::sum_of_squares_result<T>
vector_sum_of_squares_gold_solution(x_t x,
				    std::experimental::linalg::sum_of_squares_result<T> init)
{
  using std::abs;

  T scale = init.scaling_factor;
  for (std::size_t i = 0; i < x.extent(0); ++i) {
    scale = std::max(scale, abs(x(i)));
  }

  T ssq = (init.scaling_factor*init.scaling_factor*init.scaled_sum_of_squares)/(scale*scale);
  T s=0.;
  for (std::size_t i = 0; i < x.extent(0); ++i) {
    const auto absxi = abs(x(i));
    const auto quotient = absxi/scale;
    ssq = ssq + quotient * quotient;
    s += absxi*absxi;
  }

  std::experimental::linalg::sum_of_squares_result<T> result;
  result.scaled_sum_of_squares = ssq;
  result.scaling_factor = scale;

  // verify that things are consistent according to definition
  // scaled_sum_of_squares: is a value such that
  // scaling_factor^2 * scaled_sum_of_squares equals the
  // sum of squares of abs(x[i]) plus init.scaling_factor^2 * init.scaled_sum_of_squares.
  //
  const auto lhs = scale*scale*ssq;
  const auto rhs = s+init.scaling_factor*init.scaling_factor*init.scaled_sum_of_squares;
  std::cout << "Gold check : " << lhs << " " << rhs << std::endl;
  if constexpr(std::is_same_v<T, float>){
    EXPECT_NEAR(lhs, rhs, 1e-2);
  }
  if constexpr(std::is_same_v<T, double>){
    EXPECT_NEAR(lhs, rhs, 1e-9);
  }

  return result;
}

template<class x_t, class T>
void kokkos_blas1_vector_sum_of_squares_test_impl(x_t x,
						  std::experimental::linalg::sum_of_squares_result<T> initValue)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x to verify they are not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);

  const auto gold = vector_sum_of_squares_gold_solution(x, initValue);
  auto result = stdla::vector_sum_of_squares(KokkosKernelsSTD::kokkos_exec<>(),
					  x, initValue);

  if constexpr(std::is_same_v<T, float>)
  {
    EXPECT_NEAR(result.scaled_sum_of_squares, gold.scaled_sum_of_squares, 1e-3);
    EXPECT_NEAR(result.scaling_factor,	      gold.scaling_factor,	  1e-3);
  }

  if constexpr(std::is_same_v<T, double>)
  {
    EXPECT_NEAR(result.scaled_sum_of_squares, gold.scaled_sum_of_squares, 1e-9);
    EXPECT_NEAR(result.scaling_factor,	      gold.scaling_factor,	  1e-9);
  }

  // x should not change after kernel
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i) == x_preKernel[i]);
  }

}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, kokkos_vector_sum_of_squares)
{
  namespace stdla = std::experimental::linalg;
  stdla::sum_of_squares_result<value_type> init_value{2.5f, 1.2f};
  kokkos_blas1_vector_sum_of_squares_test_impl(x, init_value);
}

TEST_F(blas1_signed_double_fixture, kokkos_vector_sum_of_squares)
{
  namespace stdla = std::experimental::linalg;
  stdla::sum_of_squares_result<value_type> init_value{3.0, 1.2};
  kokkos_blas1_vector_sum_of_squares_test_impl(x, init_value);
}

TEST_F(blas1_signed_complex_double_fixture, kokkos_vector_sum_of_squares)
{
  namespace stdla = std::experimental::linalg;
  stdla::sum_of_squares_result<double> init_value{2.5, 1.2};
  kokkos_blas1_vector_sum_of_squares_test_impl(x, init_value);
}
