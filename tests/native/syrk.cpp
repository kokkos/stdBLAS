#include "./gtest_fixtures.hpp"

#include <experimental/linalg>
#include <array>

namespace {
  using std::experimental::linalg::symmetric_matrix_rank_k_update;
  using std::experimental::linalg::transposed;
  using std::experimental::linalg::upper_triangle;
  using std::extents;
  using std::layout_left;
  using std::mdspan;

  // Regression test for https://github.com/kokkos/stdBLAS/issues/261
  //
  // The reference implementation needs to implement all constraints
  // of symmetric_matrix_rank_k_update in order to disambiguate
  // overloads.
  TEST(BLAS3_syrk, Issue261)
  {
    constexpr auto map_C = layout_left::mapping{extents<std::size_t,3,3>{}};
    constexpr auto map_expected = map_C;

    // [1.0   2.0   3.0]
    // [2.0   4.0   5.0]
    // [3.0   5.0   6.0]
    std::array<double, 9> WC{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
    mdspan C{WC.data(), map_C};

    // [1.0   1.0   2.0]
    // [      1.0   2.0]
    // [            2.0]
    std::array<double, 6> WA{1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    mdspan A{WA.data(), layout_left::mapping{extents<std::size_t,3,2>{}}};

    // [7.0   7.0   7.0]
    // [7.0   9.0   9.0]
    // [7.0   9.0  10.0]
    std::array<double, 9> expected_storage{7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 7.0, 9.0, 10.0};
    mdspan expected{expected_storage.data(), map_expected};

    symmetric_matrix_rank_k_update(1.0, A, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
        const auto expected_rc = expected(row, col);
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc);
      }
    }
  }

} // end anonymous namespace
