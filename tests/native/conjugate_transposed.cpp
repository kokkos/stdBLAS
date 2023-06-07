#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <complex>
#include <vector>

namespace {
  using std::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::conjugate_transposed;

  TEST(conjugate_transposed, mdspan_complex_double)
  {
    using std::conj;
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using matrix_dynamic_t =
      mdspan<scalar_t, extents<std::size_t, dynamic_extent, dynamic_extent>>;
    constexpr std::size_t dim = 5;
    using matrix_static_t =
      mdspan<scalar_t, extents<std::size_t, dim, dim>>;

    constexpr std::size_t storageSize = std::size_t(dim*dim);
    std::vector<scalar_t> A_storage (storageSize);
    std::vector<scalar_t> B_storage (storageSize);

    matrix_dynamic_t A (A_storage.data (), dim, dim);
    matrix_static_t B (B_storage.data ());

    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        const real_t i_val_re (real_t(i) + 1.0);
        const scalar_t i_val (i_val_re, i_val_re);
        const real_t j_val_re = real_t(j) + 1.0;
        const scalar_t j_val (j_val_re, j_val_re);
        const scalar_t val = i_val + real_t(dim) * j_val;

        A[i,j] = val;
        B[i,j] = -val;
      }
    }

    auto A_h = conjugate_transposed (A);
    auto B_h = conjugate_transposed (B);

    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        const real_t i_val_re (real_t(i) + 1.0);
        const scalar_t i_val (i_val_re, i_val_re);
        const real_t j_val_re = real_t(j) + 1.0;
        const scalar_t j_val (j_val_re, j_val_re);
        const scalar_t val = i_val + real_t(dim) * j_val;

        // AMK 5.6.23 googletest gets confused by the [r,c] notation
        // and gives an error message about having 3 params instead of 2
        EXPECT_EQ( (A[i,j]), val );
        EXPECT_EQ( (B[i,j]), -val );

        EXPECT_EQ( scalar_t(A_h[j,i]), conj(val) );
        EXPECT_EQ( scalar_t(B_h[j,i]), -conj(val) );

        EXPECT_EQ( scalar_t(A_h[j,i]), conj(A[i,j]) );
        EXPECT_EQ( scalar_t(B_h[j,i]), conj(B[i,j]) );
      }
    }
  }
}
