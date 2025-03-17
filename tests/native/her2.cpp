#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::hermitian_matrix_rank_2_update;
  using LinearAlgebra::transposed;
  using LinearAlgebra::scaled;
  using LinearAlgebra::upper_triangle;

  //     [1000.0  2000.0  3000.0]
  // A = [***     4000.0  5000.0]
  //     [***     ***     7000.0]
  //
  //         [3.0 ]   [11.0 13.0 -17.0i]   [33.0   39.0   -51.0i]
  // x y^H = [5.0i] *                    = [55.0i  65.0i   85.0 ]
  //         [7.0 ]                        [77.0   91.0  -119.0i]
  //
  //         [11.0 ]   [3.0 -5.0i 7.0]     [33.0  -55.0i  77.0 ]
  // y x^H = [13.0 ] *                   = [39.0  -65.0i  91.0 ]
  //         [17.0i]                       [51.0i  85.0  119.0i]
  //
  // x y^T + y x^T = [66       39-55i   77-51i]
  //                 [39+55i    0      176    ]
  //                 [77+51i  176       0     ]
  //
  // [1000 2000 3000]   [66       39-55i   77-51i]   [1066 2039-55i 3077-51i]
  // [**** 4000 5000] + [39+55i    0      176    ] = [**** 4000     5176    ]
  // [**** **** 7000]   [77+51i  176       0     ]   [**** ****     7000    ]
  TEST(BLAS3_syrk, Test0)
  {
    constexpr auto map_A = layout_left::mapping{extents<std::size_t,3,3>{}};
    constexpr auto map_x = layout_right::mapping{extents<std::size_t,3>{}};
    constexpr auto map_y = map_x;

    constexpr std::array<std::complex<double>, 3> x_storage_original{
      std::complex{ 3.0, 0.0},
      std::complex{ 0.0, 5.0},
      std::complex{7.0,  0.0}
    };
    constexpr std::array<std::complex<double>, 3> y_storage_original{
      std::complex{11.0, 0.0},
      std::complex{13.0, 0.0},
      std::complex{0.0, 17.0}
    };
    // Fill in the "empty spots" with a large negative value.
    // The algorithm should never see it, but if it does,
    // then the results will be wrong in an obvious way.
    constexpr std::complex<double> flag{-20000.0, -20000.0};
    constexpr std::array<std::complex<double>, 9> A_storage_original{
      std::complex{1000.0, 0.0},                      flag,          flag,
      std::complex{2000.0, 0.0}, std::complex{4000.0, 0.0},          flag,
      std::complex{3000.0, 0.0}, std::complex{5000.0, 0.0}, std::complex{7000.0, 0.0}
    };

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      constexpr std::array<std::complex<double>, 3> x_storage = x_storage_original;
      constexpr std::array<std::complex<double>, 3> y_storage = y_storage_original;
      std::array<std::complex<double>, 9> result_storage{};

      mdspan x{x_storage.data(), map_x};
      mdspan y{y_storage.data(), map_y};
      mdspan result{result_storage.data(), map_A};

      // result := x y^T + y x^T
      hermitian_matrix_rank_2_update(x, y, result, upper_triangle);

      // x y^T + y x^T = [66       39-55i   77-51i]
      //                 [39+55i    0      176    ]
      //                 [77+51i  176       0     ]
      constexpr std::array<std::complex<double>, 9> expected_storage{
        std::complex{66.0,   0.0}, std::complex{ 39.0, 55.0}, std::complex{ 77.0, 51.0},
        std::complex{39.0, -55.0}, std::complex{  0.0,  0.0}, std::complex{176.0,  0.0},
        std::complex{77.0, -51.0}, std::complex{176.0,  0.0}, std::complex{  0.0,  0.0}
      };
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
      constexpr std::array<std::complex<double>, 3> x_storage = x_storage_original;
      constexpr std::array<std::complex<double>, 3> y_storage = y_storage_original;
      constexpr std::array<std::complex<double>, 9> A_storage = A_storage_original;
      std::array<std::complex<double>, 9> result_storage{};

      mdspan A{A_storage.data(), map_A};
      mdspan x{x_storage.data(), map_x};
      mdspan y{y_storage.data(), map_y};
      mdspan result{result_storage.data(), map_A};

      // result := x y^T + y x^T + A
#if defined(LINALG_FIX_RANK_UPDATES)
      hermitian_matrix_rank_2_update(x, y, A, result, upper_triangle);
#else
      for (std::size_t r = 0; r < A.extent(0); ++r) {
        for (std::size_t c = 0; c < A.extent(0); ++c) {
          result(r, c) = A(r, c);
        }
      }
      hermitian_matrix_rank_2_update(x, y, result, upper_triangle);
#endif
      // [1000 2000 3000]   [66       39-55i   77-51i]   [1066 2039-55i 3077-51i]
      // [**** 4000 5000] + [39+55i    0      176    ] = [**** 4000     5176    ]
      // [**** **** 7000]   [77+51i  176       0     ]   [**** ****     7000    ]
      constexpr std::array<std::complex<double>, 9> expected_storage{
        std::complex{1066.0,   0.0},                      flag,                      flag,
        std::complex{2039.0, -55.0}, std::complex{4000.0, 0.0},                      flag,
        std::complex{3077.0, -51.0}, std::complex{5176.0, 0.0}, std::complex{7000.0, 0.0}
      };
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
