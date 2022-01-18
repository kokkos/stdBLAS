
#include "gtest_fixtures.hpp"
#include "helpers.hpp"
#include <cmath>

namespace
{

template<class A_t, class T>
T matrix_frob_norm_gold_solution(A_t A, T initValue, bool useInit)
{
  using std::abs;
  using value_type = typename A_t::value_type;

  T result = {};
  for (std::size_t i=0; i<A.extent(0); ++i){
    for (std::size_t j=0; j<A.extent(1); ++j){
      if constexpr(std::is_same_v<value_type, std::complex<double>>){
	result += std::norm(A(i,j));
      }
      else{
	result += abs(A(i,j)) * abs(A(i,j));
      }
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

template<class A_t, class T>
void kokkos_matrix_frob_norm_test_impl(A_t A, T initValue, bool useInit)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // copy x to verify they are not changed after kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);

  const T gold = matrix_frob_norm_gold_solution(A, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(),
				     A, initValue);
  }else{
    result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(),
				     A);
  }

  // need to fix tolerances
  if constexpr(std::is_same_v<value_type, float>){
    EXPECT_NEAR(result, gold, 1e-2);
  }

  if constexpr(std::is_same_v<value_type, double>){
    EXPECT_NEAR(result, gold, 1e-9);
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    EXPECT_NEAR(result, gold, 1e-9);
  }

  // A should not change after kernel
  std::size_t k=0;
  for (std::size_t i=0; i<extent0; ++i){
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_TRUE(A(i,j) == A_preKernel[k++]);
    }
  }

}
}//end anonym namespace

//
// float
//
TEST_F(blas2_signed_float_fixture, kokkos_matrix_frob_norm_trivial_empty)
{
  std::vector<value_type> v;

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 0, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_frob_norm_trivial_one_element)
{
  constexpr value_type myvalue = static_cast<value_type>(-1.5);
  std::vector<value_type> v = {myvalue};

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 1, 1);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{} + std::abs(myvalue));
  }
  {
    constexpr value_type init = static_cast<value_type>(2.8);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init + std::abs(myvalue));
  }
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_frob_norm_noinitvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<value_type>(0), false);
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_frob_norm_initvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<value_type>(5.6), true);
}

//
// double
//
TEST_F(blas2_signed_double_fixture, kokkos_matrix_frob_norm_trivial_empty)
{
  std::vector<value_type> v;

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 0, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_frob_norm_trivial_one_element)
{
  constexpr value_type myvalue = static_cast<value_type>(-1.5);
  std::vector<value_type> v = {myvalue};

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 1, 1);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{} + std::abs(myvalue));
  }
  {
    constexpr value_type init = static_cast<value_type>(2.8);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init + std::abs(myvalue));
  }
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_frob_norm_noinitvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<value_type>(0), false);
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_frob_norm_initvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<value_type>(5), true);
}

//
// complex double
//
TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_frob_norm_trivial_empty)
{
  std::vector<value_type> v;

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 0, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr double init = static_cast<double>(2.5);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_frob_norm_trivial_one_element)
{
  constexpr value_type myvalue{-1.5, 2.2};
  std::vector<value_type> v = {myvalue};

  constexpr auto de = std::experimental::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 1, 1);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype( std::abs(M(0,0)) * std::abs(M(0,0)) );
    EXPECT_TRUE(result == init_t{} + std::abs(myvalue));
  }
  {
    constexpr double init = static_cast<double>(2.5);
    const auto result = stdla::matrix_frob_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init + std::abs(myvalue));
  }
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_frob_norm_noinitvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<double>(0), false);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_frob_norm_initvalue)
{
  kokkos_matrix_frob_norm_test_impl(A, static_cast<double>(5), true);
}
