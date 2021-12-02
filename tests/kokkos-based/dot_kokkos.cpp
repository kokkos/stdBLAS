
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class T>
void kokkos_blas1_dot_verify(const T r1, const T r2)
{
  if constexpr(std::is_same_v<T, double>){
    // I have to do this because I noticed in some cases DOUBLE_EQ
    // gives troubles
    EXPECT_NEAR(r1, r2, 1e-9);
  }
}

template<class x_t, class y_t, class T>
void kokkos_blas1_dot_test_impl(x_t x, y_t y,
				 T factorX, T factorY,
				 bool scaleX, bool scaleY,
				 T initValue,
				 bool useInit)
{
  namespace stdla = std::experimental::linalg;

  const std::size_t extent = x.extent(0);

  // copy x and y to verify they are not changed after kernel
  auto x_gold = kokkostesting::create_stdvector_and_copy(x);
  auto y_gold = kokkostesting::create_stdvector_and_copy(y);

  // use sequential as the gold
  std::vector<T> gold(extent);
  using mdspan_t = mdspan<T, extents<dynamic_extent>>;
  mdspan_t z_gold(gold.data(), extent);

  if (!scaleX && !scaleY){
    const auto r1 = stdla::dot(std::execution::seq,		  x, y);
    const auto r2 = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(), x, y);
    kokkos_blas1_dot_verify(r1, r2);
  }

  if (scaleX && !scaleY){
    const auto r1 = stdla::dot(std::execution::seq,
			       stdla::scaled(factorX, x),
			       y);
    const auto r2 = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(),
			       stdla::scaled(factorX, x),
			       y);
    kokkos_blas1_dot_verify(r1, r2);
  }

  if (!scaleX && scaleY){
    const auto r1 = stdla::dot(std::execution::seq,
			       x,
			       stdla::scaled(factorY, y));
    const auto r2 = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(),
			       x,
			       stdla::scaled(factorY, y));
    kokkos_blas1_dot_verify(r1, r2);
  }

  if (scaleX && scaleY){
    const auto r1 = stdla::dot(std::execution::seq,
			       stdla::scaled(factorX, x),
			       stdla::scaled(factorY, y));
    const auto r2 = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(),
			       stdla::scaled(factorX, x),
			       stdla::scaled(factorY, y));
    kokkos_blas1_dot_verify(r1, r2);
  }

  if constexpr(std::is_same_v<T, double>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_gold[i]);
      EXPECT_DOUBLE_EQ(y(i), y_gold[i]);
    }
  }
}
}//end anonym namespace


TEST_F(blas1_signed_double_fixture, kokkos_dot_noinitvalue)
{
  kokkos_blas1_dot_test_impl(x, y, 0., 0., false, false, 0., false);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_noinitvalue_scaled_accessor_x)
{
  kokkos_blas1_dot_test_impl(x, y, 2., 0., true, false, 0., false);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_noinitvalue_scaled_accessor_y)
{
  kokkos_blas1_dot_test_impl(x, y, 0., 2., false, true, 0., false);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_noinitvalue_scaled_accessor_xy)
{
  kokkos_blas1_dot_test_impl(x, y, 2., 3., true, true, 0., false);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_initvalue)
{
  kokkos_blas1_dot_test_impl(x, y, 0., 0., false, false, 5., true);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_initvalue_scaled_accessor_x)
{
  kokkos_blas1_dot_test_impl(x, y, 2., 0., true, false, 5., true);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_initvalue_scaled_accessor_y)
{
  kokkos_blas1_dot_test_impl(x, y, 0., 2., false, true, 5., true);
}

TEST_F(blas1_signed_double_fixture, kokkos_dot_initvalue_scaled_accessor_xy)
{
  kokkos_blas1_dot_test_impl(x, y, 3., 2., true, true, 5., true);
}
