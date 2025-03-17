#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::symmetric_matrix_rank_2_update;
  using LinearAlgebra::transposed;
  using LinearAlgebra::scaled;
  using LinearAlgebra::upper_triangle;

  //     [1000.0  2000.0  3000.0]
  // A = [***     4000.0  5000.0]
  //     [***     ***     7000.0]
  //
  //         [3.0]   [11.0 13.0 17.0]   [33.0  39.0  51.0]
  // x y^T = [5.0] *                  = [55.0  65.0  85.0]
  //         [7.0]                      [77.0  91.0 119.0]
  //
  //         [11.0]   [3.0 5.0 7.0]     [33.0  55.0  77.0]
  // y x^T = [13.0] *                 = [39.0  65.0  91.0]
  //         [17.0]                     [51.0  85.0 119.0]
  //
  // x y^T + y x^T = [ 66.0   94.0  128.0]
  //                 [ 94.0  130.0  176.0]
  //                 [128.0  176.0  238.0]
  //
  // [1000.0 2000.0 3000.0]   [ 66.0   94.0  128.0]   [1066.0 2094.0 3128.0]
  // [****** 4000.0 5000.0] + [ 94.0  130.0  176.0] = [****** 4130.0 5176.0]
  // [****** ****** 7000.0]   [128.0  176.0  238.0]   [****** ****** 7238.0]
  //
  // [1000.0 2000.0 3000.0]         [ 66.0   94.0  128.0]   [1132.0 2188.0 3256.0]
  // [****** 4000.0 5000.0] + 2.0 * [ 94.0  130.0  176.0] = [****** 4260.0 5352.0]
  // [****** ****** 7000.0]         [128.0  176.0  238.0]   [****** ****** 7476.0]
  TEST(BLAS3_syrk, Test0)
  {
    constexpr auto map_A = layout_left::mapping{extents<std::size_t,3,3>{}};
    constexpr auto map_x = layout_right::mapping{extents<std::size_t,3>{}};
    constexpr auto map_y = map_x;

    constexpr std::array<double, 3> x_storage_original{
      3.0, 5.0, 7.0};
    constexpr std::array<double, 3> y_storage_original{
      11.0, 13.0, 17.0};
    // Fill in the "empty spots" with a large negative value.
    // The algorithm should never see it, but if it does,
    // then the results will be wrong in an obvious way.
    constexpr double flag = -1000.0;
    constexpr std::array<double, 9> A_storage_original{
      1000.0, flag, flag, 2000.0, 4000.0, flag, 3000.0, 5000.0, 7000.0};

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      constexpr std::array<double, 3> x_storage = x_storage_original;
      constexpr std::array<double, 3> y_storage = y_storage_original;
      std::array<double, 9> result_storage{};

      mdspan x{x_storage.data(), map_x};
      mdspan y{y_storage.data(), map_y};
      mdspan result{result_storage.data(), map_A};

      // result := x y^T + y x^T
      symmetric_matrix_rank_2_update(x, y, result, upper_triangle);

      // [ 66.0   94.0  128.0]
      // [ 94.0  130.0  176.0]
      // [128.0  176.0  238.0]
      constexpr std::array<double, 9> expected_storage{
        66.0, 94.0, 128.0, 94.0, 130.0, 176.0, 128.0, 176.0, 238.0};
      mdspan expected{expected_storage.data(), map_A};
      for (std::size_t row = 0; row < result.extent(0); ++row) {
        for (std::size_t col = row; col < result.extent(1); ++col) {
          const auto expected_rc = expected(row, col);
          const auto result_rc = result(row, col);
          EXPECT_EQ(expected_rc, result_rc) << "at (" << row << ", " << col << ")";
        }
      }
    }
#endif
    {
      constexpr std::array<double, 3> x_storage = x_storage_original;
      constexpr std::array<double, 3> y_storage = y_storage_original;
      constexpr std::array<double, 9> A_storage = A_storage_original;
      std::array<double, 9> result_storage{};

      mdspan A{A_storage.data(), map_A};
      mdspan x{x_storage.data(), map_x};
      mdspan y{y_storage.data(), map_y};
      mdspan result{result_storage.data(), map_A};

      // result := x y^T + y x^T + A
#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_2_update(x, y, A, result, upper_triangle);
#else
      for (std::size_t r = 0; r < A.extent(0); ++r) {
        for (std::size_t c = 0; c < A.extent(0); ++c) {
          result(r, c) = A(r, c);
        }
      }
      symmetric_matrix_rank_2_update(x, y, result, upper_triangle);
#endif
      // [1000.0 2000.0 3000.0]   [ 66.0   94.0  128.0]   [1066.0 2094.0 3128.0]
      // [****** 4000.0 5000.0] + [ 94.0  130.0  176.0] = [****** 4130.0 5176.0]
      // [****** ****** 7000.0]   [128.0  176.0  238.0]   [****** ****** 7238.0]
      constexpr std::array<double, 9> expected_storage{
        1066.0, flag, flag, 2094.0, 4130.0, flag, 3128.0, 5176.0, 7238.0};
      mdspan expected{expected_storage.data(), map_A};
      for (std::size_t row = 0; row < result.extent(0); ++row) {
        for (std::size_t col = row; col < result.extent(1); ++col) {
          const auto expected_rc = expected(row, col);
          const auto result_rc = result(row, col);
          EXPECT_EQ(expected_rc, result_rc) << "at (" << row << ", " << col << ")";
        }
      }
    }
    {
      constexpr std::array<double, 3> x_storage = x_storage_original;
      constexpr std::array<double, 3> y_storage = y_storage_original;
      constexpr std::array<double, 9> A_storage = A_storage_original;
      std::array<double, 9> result_storage{};

      mdspan A{A_storage.data(), map_A};
      mdspan x{x_storage.data(), map_x};
      mdspan y{y_storage.data(), map_y};
      mdspan result{result_storage.data(), map_A};

      // result := 2.0 (x y^T + y x^T) + A
#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_2_update(scaled(2.0, x), y, A, result, upper_triangle);
#else
      for (std::size_t r = 0; r < A.extent(0); ++r) {
        for (std::size_t c = 0; c < A.extent(0); ++c) {
          result(r, c) = A(r, c);
        }
      }
      symmetric_matrix_rank_2_update(scaled(2.0, x), y, result, upper_triangle);
#endif
      // [1000.0 2000.0 3000.0]         [ 66.0   94.0  128.0]   [1132.0 2188.0 3256.0]
      // [****** 4000.0 5000.0] + 2.0 * [ 94.0  130.0  176.0] = [****** 4260.0 5352.0]
      // [****** ****** 7000.0]         [128.0  176.0  238.0]   [****** ****** 7476.0]
      constexpr std::array<double, 9> expected_storage{
        1132.0, flag, flag, 2188.0, 4260.0, flag, 3256.0, 5352.0, 7476.0};
      mdspan expected{expected_storage.data(), map_A};
      for (std::size_t row = 0; row < result.extent(0); ++row) {
        for (std::size_t col = row; col < result.extent(1); ++col) {
          const auto expected_rc = expected(row, col);
          const auto result_rc = result(row, col);
          EXPECT_EQ(expected_rc, result_rc) << "at (" << row << ", " << col << ")";
        }
      }
    }
  }
} // end anonymous namespace
