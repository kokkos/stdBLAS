#include "./gtest_fixtures.hpp"

#include <experimental/linalg>
#include <array>

namespace {
  using std::experimental::linalg::lower_triangle;
  using std::experimental::linalg::symmetric_matrix_rank_1_update;
  using std::experimental::linalg::upper_triangle;
  using std::extents;
  using std::layout_right;
  using std::mdspan;

  // Regression test for ambiguous overloads of
  // symmetric_matrix_rank_1_update (related to
  // https://github.com/kokkos/stdBLAS/issues/261).
  //
  // The reference implementation needs to implement all constraints
  // of symmetric_matrix_rank_1_update in order to disambiguate
  // overloads.
  TEST(BLAS3_syr, AmbiguousOverloads)
  {
    constexpr auto map_A = layout_right::mapping{extents<std::size_t, 3, 3>{}};
    constexpr auto map_expected = map_A;
    constexpr auto map_x = layout_right::mapping{extents<std::size_t, 3>{}};

    // A = [-1.0  -2.0  -4.0]
    //     [-2.0  -3.0  -5.0]
    //     [-4.0  -5.0  -6.0]
    //
    // x = [  2.0]
    //     [  5.0]
    //     [ 11.0]
    //
    // x x^T = [4.0   10.0  22.0]
    //         [10.0  25.0  55.0]
    //         [22.0  55.0 121.0]
    //
    // A + x x^T = [3.0   8.0  18.0]
    //             [8.0  22.0  50.0]
    //             [18.0 50.0 115.0]
    constexpr std::array<double, 9> A_storage_original{
      -1.0, -2.0, -4.0,
      -2.0, -3.0, -5.0,
      -4.0, -5.0, -6.0
    };
    constexpr std::array<double, 3> x_storage_original{
      2.0,
      5.0,
      11.0
    };
    constexpr std::array<double, 9> expected_storage_original{
      3.0,  8.0,  18.0,
      8.0,  22.0, 50.0,
      18.0, 50.0, 115.0
    };

    auto A_storage = A_storage_original;
    mdspan A{A_storage.data(), map_A};

    auto expected_storage = expected_storage_original;
    mdspan expected{expected_storage.data(), map_expected};

    auto x_storage = x_storage_original;
    mdspan x{x_storage.data(), map_x};

    auto check_upper_triangle = [&] () {
      for (std::size_t row = 0; row < A.extent(0); ++row) {
        for (std::size_t col = row; col < A.extent(1); ++col) {
          const auto expected_rc = expected(row, col);
          const auto A_rc = A(row, col);
          EXPECT_EQ(expected_rc, A_rc) << "at (" << row << ", " << col << ")";
        }
      }
    };
    auto check_lower_triangle = [&] () {
      for (std::size_t row = 0; row < A.extent(0); ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
          const auto expected_rc = expected(row, col);
          const auto A_rc = A(row, col);
          EXPECT_EQ(expected_rc, A_rc) << "at (" << row << ", " << col << ")";
        }
      }
    };

    symmetric_matrix_rank_1_update(1.0, x, A, upper_triangle);
    check_upper_triangle();

    // Reset values, just in case some bug might have overwritten them.
    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    symmetric_matrix_rank_1_update(1.0, x, A, lower_triangle);
    check_lower_triangle();

    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    symmetric_matrix_rank_1_update(x, A, upper_triangle);
    check_upper_triangle();

    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    symmetric_matrix_rank_1_update(x, A, lower_triangle);
    check_lower_triangle();
  }

} // end anonymous namespace
