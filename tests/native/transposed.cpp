#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <vector>

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::transposed;

  template<std::size_t ext0, std::size_t ext1>
  void test_transpose_extents()
  {
    using std::experimental::linalg::impl::transpose_extents_t;
    using std::experimental::linalg::impl::transpose_extents;

    using extents_type = extents<ext0, ext1>;
    using expected_transpose_extents_type = extents<ext1, ext0>;
    using transpose_extents_type = transpose_extents_t<extents_type>;
    static_assert(std::is_same_v<expected_transpose_extents_type, transpose_extents_type>);

    using size_type = typename extents_type::size_type;
    constexpr size_type numRows = 666;
    constexpr size_type numCols = 777;

    if constexpr (ext0 == dynamic_extent) {
      if constexpr (ext1 == dynamic_extent) {
	extents_type orig(numRows, numCols);
	auto xpose = transpose_extents(orig);
	static_assert(std::is_same_v<expected_transpose_extents_type, decltype(xpose)>);
	EXPECT_EQ(orig.extent(0), xpose.extent(1));
	EXPECT_EQ(orig.extent(1), xpose.extent(0));
      } else {
	extents_type orig(numRows);
	auto xpose = transpose_extents(orig);
	static_assert(std::is_same_v<expected_transpose_extents_type, decltype(xpose)>);
	EXPECT_EQ(orig.extent(0), xpose.extent(1));
	EXPECT_EQ(orig.extent(1), xpose.extent(0));
      }
    } else {
      if constexpr (ext1 == dynamic_extent) {
	extents_type orig(numCols);
	auto xpose = transpose_extents(orig);
	static_assert(std::is_same_v<expected_transpose_extents_type, decltype(xpose)>);
	EXPECT_EQ(orig.extent(0), xpose.extent(1));
	EXPECT_EQ(orig.extent(1), xpose.extent(0));
      } else {
	extents_type orig{};
	auto xpose = transpose_extents(orig);
	static_assert(std::is_same_v<expected_transpose_extents_type, decltype(xpose)>);
	EXPECT_EQ(orig.extent(0), xpose.extent(1));
	EXPECT_EQ(orig.extent(1), xpose.extent(0));
      }
    }
  }

  TEST(transpose_extents, test0)
  {
    test_transpose_extents<3, 3>();
    test_transpose_extents<3, 4>();
    test_transpose_extents<4, 3>();
    test_transpose_extents<dynamic_extent, 3>();
    test_transpose_extents<3, dynamic_extent>();
    test_transpose_extents<dynamic_extent, dynamic_extent>();
  }

  template<std::size_t ext0, std::size_t ext1>
  void test_layout_transpose()
  {
    using std::experimental::linalg::layout_transpose;
    using std::experimental::layout_left;
    using extents_type = extents<ext0, ext1>;
    using mapping_type = typename layout_transpose<layout_left>::mapping<extents_type>;
  }

  TEST(layout_transpose, test0)
  {
    test_layout_transpose<3, 3>();
    test_layout_transpose<3, 4>();
    test_layout_transpose<4, 3>();
    test_layout_transpose<dynamic_extent, 3>();
    test_layout_transpose<3, dynamic_extent>();
    test_layout_transpose<dynamic_extent, dynamic_extent>();
  }

  TEST(transposed, mdspan_double)
  {
    using real_t = double;
    using scalar_t = double;
    using matrix_dynamic_t =
      mdspan<scalar_t, extents<dynamic_extent, dynamic_extent>>;
    constexpr std::size_t dim = 5;
    using matrix_static_t =
      mdspan<scalar_t, extents<dim, dim>>;

    constexpr std::size_t storageSize = std::size_t(dim*dim);
    std::vector<scalar_t> A_storage (storageSize);
    std::vector<scalar_t> B_storage (storageSize);

    matrix_dynamic_t A (A_storage.data (), dim, dim);
    matrix_static_t B (B_storage.data ());

    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        const scalar_t i_val = scalar_t(i) + 1.0;
        // If we generalize this test so scalar_t can be complex, then
        // we'll need the intermediate std::size_t -> real_t conversion.
        const scalar_t j_val = scalar_t(real_t(dim)) * (scalar_t(j) + 1.0);
        const scalar_t val = i_val + j_val;

        A(i,j) = val;
        B(i,j) = -val;
      }
    }

    auto A_t = transposed (A);
    auto B_t = transposed (B);

    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        const scalar_t i_val = scalar_t(i) + 1.0;
        // If we generalize this test so scalar_t can be complex, then
        // we'll need the intermediate ptrdiff_t -> real_t conversion.
        const scalar_t j_val = scalar_t(real_t(dim)) * (scalar_t(j) + 1.0);
        const scalar_t val = i_val + j_val;

        EXPECT_EQ( A(i,j), val );
        EXPECT_EQ( B(i,j), -val );

        EXPECT_EQ( A_t(j,i), val );
        EXPECT_EQ( B_t(j,i), -val );

        EXPECT_EQ( A_t(j,i), A(i,j) );
        EXPECT_EQ( B_t(j,i), B(i,j) );
      }
    }
  }
}
