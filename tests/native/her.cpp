#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::hermitian_matrix_rank_1_update;
  using LinearAlgebra::upper_triangle;

  // Regression test for ambiguous overloads of
  // hermitian_matrix_rank_1_update (related to
  // https://github.com/kokkos/stdBLAS/issues/261 ).
  //
  // The reference implementation needs to implement all constraints
  // of hermitian_matrix_rank_1_update in order to disambiguate
  // overloads.
  TEST(BLAS3_her, AmbiguousOverloads)
  {
    constexpr auto map_A = layout_right::mapping{extents<std::size_t, 3, 3>{}};
    constexpr auto map_expected = map_A;
    constexpr auto map_x = layout_right::mapping{extents<std::size_t, 3>{}};
    using V = std::complex<double>;

    // A = [-1.0  -2.0  -4.0]
    //     [-2.0  -3.0  -5.0]
    //     [-4.0  -5.0  -6.0]
    //
    // x = [  2.0 + 7.0i]
    //     [  5.0]
    //     [ 11.0]
    //
    // x x^H = [53.0          10.0 + 35.0i   22.0 + 77.0i]
    //         [10.0 - 35.0i  25.0           55.0        ]
    //         [22.0 - 77.0i  55.0          121.0        ]
    //
    // A + x x^H = [52.0           8.0 + 35.0i   18.0 + 77.0i]
    //             [ 8.0 - 35.0i  22.0           50.0]
    //             [18.0 - 77.0i  50.0          115.0]
    constexpr std::array<V, 9> A_storage_original{
      V(-1.0, 0.0), V(-2.0, 0.0), V(-4.0, 0.0),
      V(-2.0, 0.0), V(-3.0, 0.0), V(-5.0, 0.0),
      V(-4.0, 0.0), V(-5.0, 0.0), V(-6.0, 0.0)
    };
    constexpr std::array<V, 3> x_storage_original{
      V( 2.0, 7.0),
      V( 5.0, 0.0),
      V(11.0, 0.0)
    };
    constexpr std::array<V, 9> expected_storage_original{
      V(52.0,   0.0), V( 8.0, 35.0),  V( 18.0, 77.0),
      V( 8.0, -35.0), V(22.0,  0.0),  V( 50.0,  0.0),
      V(18.0, -77.0), V(50.0,  0.0),  V(115.0,  0.0)
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

    hermitian_matrix_rank_1_update(1.0, x, A, upper_triangle);
    check_upper_triangle();

    // Reset values, just in case some bug might have overwritten them.
    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    hermitian_matrix_rank_1_update(1.0, x, A, lower_triangle);
    check_lower_triangle();

    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    hermitian_matrix_rank_1_update(x, A, upper_triangle);
    check_upper_triangle();

    A_storage = A_storage_original;
    expected_storage = expected_storage_original;
    x_storage = x_storage_original;
    hermitian_matrix_rank_1_update(x, A, lower_triangle);
    check_lower_triangle();
  }

} // end anonymous namespace
