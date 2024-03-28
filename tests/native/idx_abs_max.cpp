#include "./gtest_fixtures.hpp"

namespace {

  using LinearAlgebra::idx_abs_max;

  TEST_F(unsigned_double_vector, idx_abs_max)
  {
    constexpr size_t expected(9);
    EXPECT_EQ(expected, idx_abs_max(v));
  }

  TEST_F(signed_double_vector, idx_abs_max)
  {
    constexpr size_t expected(9);
    EXPECT_EQ(expected, idx_abs_max(v));
  }

  TEST_F(signed_complex_vector, idx_abs_max)
  {
    constexpr size_t expected(2);
    EXPECT_EQ(expected, idx_abs_max(v));
  }

  TEST(BLAS1_idx_abs_max, trivial_case)
  {
    constexpr auto expected = std::numeric_limits<std::size_t>::max();

    std::array<double, 0> arr;
    using extents_type = extents<std::size_t, dynamic_extent>;
    mdspan<double, extents_type> a(arr.data(),0);
    EXPECT_EQ(expected, idx_abs_max(a));

    using extents_type2 = extents<std::size_t, 0>;
    mdspan<double, extents_type2> b(arr.data());
    EXPECT_EQ(expected, idx_abs_max(b));
  }

} // end anonymous namespace
