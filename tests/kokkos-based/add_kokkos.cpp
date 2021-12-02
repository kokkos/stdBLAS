
#include "gtest_blas1_fixtures.hpp"
#include "helpers.hpp"

namespace{

template<class x_t, class y_t, class z_t, class T>
void kokkos_blas1_add_test_impl(x_t x, y_t y, z_t z,
				 T factorX, T factorY,
				 bool scaleX, bool scaleY)
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
    stdla::add(std::execution::seq,		  x, y, z_gold);
    stdla::add(KokkosKernelsSTD::kokkos_exec<>(), x, y, z);
  }

  if (scaleX && !scaleY){
    stdla::add(std::execution::seq,
	       stdla::scaled(factorX, x),
	       y,
	       z_gold);
    stdla::add(KokkosKernelsSTD::kokkos_exec<>(),
	       stdla::scaled(factorX, x),
	       y,
	       z);
  }

  if (!scaleX && scaleY){
    stdla::add(std::execution::seq,
	       x,
	       stdla::scaled(factorY, y),
	       z_gold);
    stdla::add(KokkosKernelsSTD::kokkos_exec<>(),
	       x,
	       stdla::scaled(factorY, y),
	       z);
  }

  if (scaleX && scaleY){
    stdla::add(std::execution::seq,
	       stdla::scaled(factorX, x),
	       stdla::scaled(factorY, y),
	       z_gold);
    stdla::add(KokkosKernelsSTD::kokkos_exec<>(),
	       stdla::scaled(factorX, x),
	       stdla::scaled(factorY, y),
	       z);
  }

  if constexpr(std::is_same_v<T, double>){
    for (std::size_t i=0; i<extent; ++i){
      EXPECT_DOUBLE_EQ(x(i), x_gold[i]);
      EXPECT_DOUBLE_EQ(y(i), y_gold[i]);
      EXPECT_DOUBLE_EQ(z(i), z_gold(i));
    }
  }
}
}//end anonym namespace


TEST_F(blas1_signed_double_fixture, kokkos_add)
{
  kokkos_blas1_add_test_impl(x, y, z, 0., 0., false, false);
}

TEST_F(blas1_signed_double_fixture, kokkos_add_with_scaled_accessor_x)
{
  kokkos_blas1_add_test_impl(x, y, z, 2., 0., true, false);
}

TEST_F(blas1_signed_double_fixture, kokkos_add_with_scaled_accessor_y)
{
  kokkos_blas1_add_test_impl(x, y, z, 0., 2., false, true);
}

TEST_F(blas1_signed_double_fixture, kokkos_add_with_scaled_accessor_xy)
{
  kokkos_blas1_add_test_impl(x, y, z, 2., 2., true, true);
}
