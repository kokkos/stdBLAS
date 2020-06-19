#include <experimental/linalg>
#include <experimental/mdspan>
#include <complex>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::conjugate_transpose_view;

  TEST(conjugate_transpose_view, mdspan_complex_double)
  {
    using std::conj;
    using real_t = double;
    using scalar_t = std::complex<real_t>;
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
        const real_t i_val_re (real_t(i) + 1.0);
        const scalar_t i_val (i_val_re, i_val_re);
        const real_t j_val_re = real_t(j) + 1.0;
        const scalar_t j_val (j_val_re, j_val_re);
        const scalar_t val = i_val + real_t(dim) * j_val;

        A(i,j) = val;
        B(i,j) = -val;
      }
    }

    auto A_h = conjugate_transpose_view (A);
    auto B_h = conjugate_transpose_view (B);

    for (ptrdiff_t i = 0; i < dim; ++i) {
      for (ptrdiff_t j = 0; j < dim; ++j) {
        const real_t i_val_re (real_t(i) + 1.0);
        const scalar_t i_val (i_val_re, i_val_re);
        const real_t j_val_re = real_t(j) + 1.0;
        const scalar_t j_val (j_val_re, j_val_re);
        const scalar_t val = i_val + real_t(dim) * j_val;

        EXPECT_EQ( A(i,j), val );
        EXPECT_EQ( B(i,j), -val );

        EXPECT_EQ( A_h(j,i), conj(val) );
        EXPECT_EQ( B_h(j,i), -conj(val) );

        EXPECT_EQ( A_h(j,i), conj(A(i,j)) );
        EXPECT_EQ( B_h(j,i), conj(B(i,j)) );
      }
    }
  }
}
