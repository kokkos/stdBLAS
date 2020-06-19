#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::transpose_view;

  TEST(transpose_view, mdspan_double)
  {
    using real_t = double;
    using scalar_t = double;
    using matrix_dynamic_t =
      basic_mdspan<scalar_t, extents<dynamic_extent, dynamic_extent>>;
    constexpr ptrdiff_t dim = 5;
    using matrix_static_t =
      basic_mdspan<scalar_t, extents<dim, dim>>;

    constexpr ptrdiff_t storageSize = ptrdiff_t(dim*dim);
    std::vector<scalar_t> A_storage (storageSize);
    std::vector<scalar_t> B_storage (storageSize);

    matrix_dynamic_t A (A_storage.data (), dim, dim);
    matrix_static_t B (B_storage.data ());

    for (ptrdiff_t i = 0; i < dim; ++i) {
      for (ptrdiff_t j = 0; j < dim; ++j) {
        const scalar_t i_val = scalar_t(i) + 1.0;
        // If we generalize this test so scalar_t can be complex, then
        // we'll need the intermediate ptrdiff_t -> real_t conversion.
        const scalar_t j_val = scalar_t(real_t(dim)) * (scalar_t(j) + 1.0);
        const scalar_t val = i_val + j_val;

        A(i,j) = val;
        B(i,j) = -val;
      }
    }

    auto A_t = transpose_view (A);
    auto B_t = transpose_view (B);

    for (ptrdiff_t i = 0; i < dim; ++i) {
      for (ptrdiff_t j = 0; j < dim; ++j) {
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
