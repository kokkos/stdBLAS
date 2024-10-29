#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::symmetric_matrix_rank_k_update;
  using LinearAlgebra::transposed;
  using LinearAlgebra::scaled;
  using LinearAlgebra::upper_triangle;

  // This is a regression test that follows on from
  // https://github.com/kokkos/stdBLAS/issues/261 .
  //
  // The reference implementation needs to implement all constraints
  // of hermitian_matrix_rank_k_update in order to disambiguate
  // overloads.
  TEST(BLAS3_herk, AmbiguousOverloads)
  {
    constexpr auto map_C = layout_left::mapping{extents<std::size_t,3,3>{}};
    constexpr auto map_expected = map_C;

    // Fill in the "empty spots" with a large negative value.
    // The algorithm should never see it, but if it does,
    // then the results will be wrong in an obvious way.
    constexpr double flag = -1000.0;
    constexpr std::array<double, 9> WA_original{
      1.0, 2.0, 3.0, 3.0, 5.0, 6.0};
    // [1.0   3.0]
    // [2.0   5.0]
    // [3.0   6.0]
    constexpr std::array<double, 9> WC_original{
      1.0, flag, flag, 3.0, 4.0, flag, 2.0, 5.0, 7.0};
    // [1.0   3.0   2.0]
    // [***   4.0   5.0]
    // [***   ***   7.0]

    // [1.0   3.0]   [1.0   2.0   3.0]   [10.0  17.0  21.0]
    // [2.0   5.0] * [3.0   5.0   6.0] = [17.0  29.0  36.0]
    // [3.0   6.0]                       [21.0  36.0  45.0]
    //
    // [1.0   3.0   2.0]   [10.0  17.0  21.0]   [11.0  20.0  23.0]
    // [***   4.0   5.0] + [17.0  29.0  36.0] = [ ***  33.0  41.0]
    // [***   ***   7.0]   [21.0  36.0  45.0]   [ ***   ***  52.0]
    constexpr std::array<double, 9> expected_storage_original{
      11.0, flag, flag, 20.0, 33.0, flag, 23.0, 41.0, 52.0};

    std::array<double, 9> WC = WC_original;
    mdspan C{WC.data(), map_C};
    mdspan C_original{WC_original.data(), map_C};

    std::array<double, 9> WA = WA_original;
    mdspan A{WA.data(), layout_left::mapping{extents<std::size_t,3,3>{}}};

    std::array<double, 9> expected_storage = expected_storage_original;
    mdspan expected{expected_storage.data(), map_expected};

    hermitian_matrix_rank_k_update(1.0, A, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
#if !defined(LINALG_FIX_RANK_UPDATES)
        const auto expected_rc = expected(row, col);
#else
        const auto expected_rc = (expected(row, col)==flag)?flag:(expected(row, col)-C_original(row, col));
#endif
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }

    // Reset values, just in case some bug might have overwritten them.
    WA = WA_original;
    WC = WC_original;
    expected_storage = expected_storage_original;

    hermitian_matrix_rank_k_update(A, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
#if !defined(LINALG_FIX_RANK_UPDATES)
        const auto expected_rc = expected(row, col);
#else
        const auto expected_rc = (expected(row, col)==flag)?flag:(expected(row, col)-C_original(row, col));
#endif
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }

#if defined(LINALG_FIX_RANK_UPDATES)

    WA = WA_original;
    WC = WC_original;
    expected_storage = expected_storage_original;

    hermitian_matrix_rank_k_update(A, C, C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
        const auto expected_rc = expected(row, col);
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }

    WA = WA_original;
    WC = WC_original;
    expected_storage = expected_storage_original;

    hermitian_matrix_rank_k_update(A, scaled(2., C), C, upper_triangle);

    for (std::size_t row = 0; row < C.extent(0); ++row) {
      for (std::size_t col = 0; col < C.extent(1); ++col) {
        const auto expected_rc = (expected(row, col)==flag)?flag:(expected(row, col)+C_original(row, col));
        const auto C_rc = C(row, col);
        EXPECT_EQ(expected_rc, C_rc) << "at (" << row << ", " << col << ")";
      }
    }
#endif
  }

} // end anonymous namespace
