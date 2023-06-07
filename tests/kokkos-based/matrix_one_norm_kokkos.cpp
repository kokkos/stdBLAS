
#include "gtest_fixtures.hpp"
#include "helpers.hpp"
#include <cmath>

namespace
{

template<class A_t, class T>
T matrix_one_norm_gold_solution(A_t A, T initValue, bool useInit)
{
  using std::abs;
  using std::max;
  using value_type = typename A_t::value_type;

  T result = abs(A(0,0));
  for (std::size_t i=1; i<A.extent(0); ++i){
    result += abs(A(i,0));
  }

  for (std::size_t j=1; j<A.extent(1); ++j){
    auto colSum = abs(A(0,j));
    for (std::size_t i=1; i<A.extent(0); ++i){
      colSum += abs(A(i,j));
    }
    result = max(colSum, result);
  }

  if (useInit){
    return initValue + result;
  }
  else{
    return result;
  }
}

template<class A_t, class T>
void kokkos_matrix_one_norm_test_impl(A_t A, T initValue, bool useInit)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // copy x to verify they are not changed after kernel
  auto A_preKernel = kokkostesting::create_stdvector_and_copy_rowwise(A);

  const T gold = matrix_one_norm_gold_solution(A, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(),
				     A, initValue);
  }else{
    result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(),
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
TEST_F(blas2_signed_float_fixture, kokkos_matrix_one_norm_trivial_empty)
{
  std::vector<value_type> v;

  constexpr auto de = std::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 0, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype(std::abs(M(0,0)));
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_one_norm_trivial_zero_col)
{
  std::vector<value_type> v{1.2, -2.4};

  constexpr auto de = std::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 2, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype(std::abs(M(0,0)));
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_one_norm_noinitvalue)
{
  kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<value_type>(0), false);
}

TEST_F(blas2_signed_float_fixture, kokkos_matrix_one_norm_initvalue)
{
  kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<value_type>(5.6), true);
}

//
// double
//
TEST_F(blas2_signed_double_fixture, kokkos_matrix_one_norm_trivial_empty)
{
  std::vector<value_type> v;

  constexpr auto de = std::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 0, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype(std::abs(M(0,0)));
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_one_norm_trivial_zero_col)
{
  std::vector<value_type> v{1.2, -2.4};

  constexpr auto de = std::dynamic_extent;
  using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
  s_t M(v.data(), 2, 0);
  namespace stdla = std::experimental::linalg;

  {
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
    using init_t = decltype(std::abs(M(0,0)));
    EXPECT_TRUE(result == init_t{});
  }
  {
    constexpr value_type init = static_cast<value_type>(2.5);
    const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
    EXPECT_TRUE(result == init);
  }
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_one_norm_noinitvalue)
{
  kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<value_type>(0), false);
}

TEST_F(blas2_signed_double_fixture, kokkos_matrix_one_norm_initvalue)
{
  kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<value_type>(5), true);
}

//
// complex double
//
TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_one_norm_trivial_empty)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){

    std::vector<value_type> v;

    constexpr auto de = std::dynamic_extent;
    using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
    s_t M(v.data(), 0, 0);
    namespace stdla = std::experimental::linalg;

    {
      const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
      using init_t = decltype(std::abs(M(0,0)));
      EXPECT_TRUE(result == init_t{});
    }
    {
      constexpr double init = static_cast<double>(2.5);
      const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
      EXPECT_TRUE(result == init);
    }
  }
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_one_norm_trivial_zero_col)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    std::vector<value_type> v(2);
    v[0] = {1.2, -1.};
    v[1] = {-2.4, 4.};

    constexpr auto de = std::dynamic_extent;
    using s_t = std::experimental::mdspan<value_type, std::experimental::extents<de, de>>;
    s_t M(v.data(), 2, 0);
    namespace stdla = std::experimental::linalg;

    {
      const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M);
      using init_t = decltype(std::abs(M(0,0)));
      EXPECT_TRUE(result == init_t{});
    }
    {
      constexpr double init = static_cast<double>(2.5);
      const auto result = stdla::matrix_one_norm(KokkosKernelsSTD::kokkos_exec<>(), M, init);
      EXPECT_TRUE(result == init);
    }
  }
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_one_norm_noinitvalue)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<double>(0), false);
  }
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_matrix_one_norm_initvalue)
{
  namespace stdla = std::experimental::linalg;
  using kc_t = Kokkos::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    kokkos_matrix_one_norm_test_impl(A_e0e1, static_cast<double>(5), true);
  }
}
