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
  TEST(BLAS3_syrk, AmbiguousOverloads_Issue261)
  {
    constexpr auto map_C = layout_left::mapping{extents<std::size_t,3,3>{}};
    constexpr auto map_expected = map_C;

    // [1.0   1.0   2.0]
    // [      1.0   2.0]
    // [            2.0]
    //
    // Fill in the "empty spots" with a large negative value.
    // The algorithm should never see it, but if it does,
    // then the results will be wrong in an obvious way.
    constexpr double flag = -1000.0;
    constexpr std::array<double, 9> WA_original{
      1.0, flag, flag, 1.0, 1.0, flag, 2.0, 2.0, 2.0};
    // [1.0   2.0   3.0]
    // [2.0   4.0   5.0]
    // [3.0   5.0   6.0]
    constexpr std::array<double, 9> WC_original{
      1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
    // [1.0   1.0   2.0]   [1.0            ]   [6.0  5.0  4.0]
    // [      1.0   2.0] * [1.0   1.0      ] = [5.0  5.0  4.0]
    // [            2.0]   [2.0   2.0   2.0]   [4.0  4.0  4.0]
    //
    // [6.0   5.0   4.0]   [1.0   2.0   3.0]   [7.0  7.0  7.0]
    // [5.0   5.0   4.0] + [2.0   4.0   5.0] = [7.0  9.0  9.0]
    // [4.0   4.0   4.0]   [3.0   5.0   6.0]   [7.0  9.0 10.0]
    constexpr std::array<double, 9> expected_storage_original{
      7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 7.0, 9.0, 10.0};

    std::array<double, 9> WC = WC_original;
    mdspan C{WC.data(), map_C};

    std::array<double, 9> WA = WA_original;
    mdspan A{WA.data(), layout_left::mapping{extents<std::size_t,3,3>{}}};

    std::array<double, 9> expected_storage = expected_storage_original;
    mdspan expected{expected_storage.data(), map_expected};

    symmetric_matrix_rank_k_update(1.0, A, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
        const auto expected_rc = expected(row, col);
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }

    // Reset values, just in case some bug might have overwritten them.
    WA = WA_original;
    WC = WC_original;
    expected_storage = expected_storage_original;

    symmetric_matrix_rank_k_update(A, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
        const auto expected_rc = expected(row, col);
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }
  }

} // end anonymous namespace
