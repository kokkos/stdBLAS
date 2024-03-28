#include "./gtest_fixtures.hpp"
#include <type_traits>

namespace {
  using LinearAlgebra::layout_transpose;
  using LinearAlgebra::transposed;
  using MdSpan::layout_left;
  using MdSpan::layout_right;
  using MdSpan::layout_stride;

  template<std::size_t ext0, std::size_t ext1>
  void test_transpose_extents()
  {
    using LinearAlgebra::impl::transpose_extents_t;
    using LinearAlgebra::impl::transpose_extents;

    using extents_type = extents<std::size_t, ext0, ext1>;
    using expected_transpose_extents_type = extents<std::size_t, ext1, ext0>;
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
    using extents_type = extents<std::size_t, ext0, ext1>;
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

  template<class InputLayoutMapping, class ExpectedLayoutMapping>
  void test_transposed_layout(const InputLayoutMapping& in,
                              const ExpectedLayoutMapping& out_expected,
                              std::vector<char>& fake_storage)
  {
    using LinearAlgebra::impl::transpose_extents_t;
    using LinearAlgebra::impl::transpose_extents;

    ASSERT_EQ(in.extents().rank(), 2u);
    ASSERT_EQ(out_expected.extents().rank(), 2u);

    const size_t required_bytes = in.required_span_size();
    if (fake_storage.size() < required_bytes) {
      fake_storage.resize(required_bytes);
    }
    mdspan in_md{fake_storage.data(), in};

    auto out_md = transposed(in_md);
    auto out = out_md.mapping();
    static_assert(std::is_same_v<decltype(out), ExpectedLayoutMapping>);
    EXPECT_EQ(out.extents(), out_expected.extents());
    EXPECT_EQ(out.is_exhaustive(), out_expected.is_exhaustive());
    EXPECT_EQ(out.is_unique(), out_expected.is_unique());
    ASSERT_EQ(out.is_strided(), out_expected.is_strided());
    if (out_expected.is_strided()) {
      for (size_t r = 0; r < 2u; ++r) {
        out.stride(r) == out_expected.stride(r);
      }
    }
  }

  TEST(transposed_layout, layout_left)
  {
    auto test_one = [] (auto in_exts, auto out_exts, std::vector<char>& fake_storage) {
      layout_left::mapping in_map{in_exts};
      layout_right::mapping out_map{out_exts};
      test_transposed_layout(in_map, out_map, fake_storage);
    };

    std::vector<char> storage;
    {
      using in_extents_type = extents<int, 3, 4>;
      using out_extents_type = extents<int, 4, 3>;
      test_one(in_extents_type{}, out_extents_type{}, storage);
    }
    {
      using in_extents_type = extents<int, 3, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, 3>;
      test_one(in_extents_type{4}, out_extents_type{4}, storage);
    }
    {
      using in_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      test_one(in_extents_type{3, 4}, out_extents_type{4, 3}, storage);
    }
  }

  TEST(transposed_layout, layout_right)
  {
    auto test_one = [] (auto in_exts, auto out_exts, std::vector<char>& fake_storage) {
      layout_right::mapping in_map{in_exts};
      layout_left::mapping out_map{out_exts};
      test_transposed_layout(in_map, out_map, fake_storage);
    };

    std::vector<char> storage;
    {
      using in_extents_type = extents<int, 3, 4>;
      using out_extents_type = extents<int, 4, 3>;
      test_one(in_extents_type{}, out_extents_type{}, storage);
    }
    {
      using in_extents_type = extents<int, 3, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, 3>;
      test_one(in_extents_type{4}, out_extents_type{4}, storage);
    }
    {
      using in_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      test_one(in_extents_type{3, 4}, out_extents_type{4, 3}, storage);
    }
  }

  TEST(transposed_layout, layout_stride)
  {
    auto test_one = [] (auto in_exts, auto out_exts, std::vector<char>& fake_storage) {
      using index_type = decltype(in_exts.extent(0));
      const std::array<index_type, 2> in_strides{
        static_cast<index_type>(2),
        static_cast<index_type>((in_exts.extent(0) + 1) * 2)
      };
      const std::array<index_type, 2> out_strides{
        in_strides[1],
        in_strides[0]
      };
      layout_stride::mapping in_map{in_exts, in_strides};
      layout_stride::mapping out_map{out_exts, out_strides};
      test_transposed_layout(in_map, out_map, fake_storage);
    };

    std::vector<char> storage;
    {
      using in_extents_type = extents<int, 3, 4>;
      using out_extents_type = extents<int, 4, 3>;
      test_one(in_extents_type{}, out_extents_type{}, storage);
    }
    {
      using in_extents_type = extents<int, 3, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, 3>;
      test_one(in_extents_type{4}, out_extents_type{4}, storage);
    }
    {
      using in_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      using out_extents_type = extents<int, dynamic_extent, dynamic_extent>;
      test_one(in_extents_type{3, 4}, out_extents_type{4, 3}, storage);
    }
  }

  TEST(transposed, mdspan_double)
  {
    using real_t = double;
    using scalar_t = double;
    using matrix_dynamic_t =
      mdspan<scalar_t, extents<std::size_t, dynamic_extent, dynamic_extent>>;
    constexpr std::size_t dim = 5;
    using matrix_static_t =
      mdspan<scalar_t, extents<std::size_t, dim, dim>>;

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
    static_assert(std::is_same_v<decltype(A)::layout_type, layout_right>);
    static_assert(std::is_same_v<decltype(A_t)::layout_type, layout_left>);
    EXPECT_EQ(A_t.extent(0), A.extent(1));
    EXPECT_EQ(A_t.extent(1), A.extent(0));

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

    constexpr std::size_t subdim = 4;
    const std::pair<std::size_t, std::size_t> sub(0, subdim);
    auto A_sub = submdspan(A, sub, sub);
    static_assert(std::is_same_v<decltype(A_sub)::layout_type, layout_stride>);
    ASSERT_EQ( A_sub.rank(), std::size_t(2) );
    ASSERT_EQ( A_sub.extent(0), subdim );
    ASSERT_EQ( A_sub.extent(1), subdim );

    auto A_sub_trans = transposed(A_sub);
    ASSERT_EQ( A_sub_trans.rank(), std::size_t(2) );
    ASSERT_EQ( A_sub_trans.extent(0), subdim );
    ASSERT_EQ( A_sub_trans.extent(1), subdim );

    for (std::size_t i = 0; i < subdim; ++i) {
      for (std::size_t j = 0; j < subdim; ++j) {
        const scalar_t i_val = scalar_t(i) + 1.0;
        // If we generalize this test so scalar_t can be complex, then
        // we'll need the intermediate ptrdiff_t -> real_t conversion.
        const scalar_t j_val = scalar_t(real_t(dim)) * (scalar_t(j) + 1.0);
        const scalar_t val = i_val + j_val;

        EXPECT_EQ( A_sub_trans(i,j), A_sub(j,i) );
        EXPECT_EQ( A_sub_trans(i,j), A(j,i) );
      }
    }
  }
}
