#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::basic_mdspan;
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::layout_blas_general;
  using std::experimental::column_major_t;  
  using std::experimental::row_major_t;

  TEST(layout_blas_general, mdspan_column_major_double)
  {
    using row_or_column_major_tag = column_major_t;
    using layout_t = layout_blas_general<row_or_column_major_tag>;
    using element_type = double;

    using dynamic_vector_t =
      basic_mdspan<element_type,
                   extents<dynamic_extent>,
                   layout_t>;
    using dynamic_matrix_t =
      basic_mdspan<element_type,
                   extents<dynamic_extent, dynamic_extent>,
                   layout_t>;
    constexpr ptrdiff_t maxDim (10);
    constexpr ptrdiff_t dim (6);
    using static_vector_t =
      basic_mdspan<element_type, extents<dim>, layout_t>;
    using static_matrix_t =
      basic_mdspan<element_type, extents<dim, dim>, layout_t>;

    constexpr size_t matrixStorageSize = maxDim * maxDim;
    std::vector<element_type> matrixStorage (matrixStorageSize);

    dynamic_matrix_t A (matrixStorage.data (), maxDim, maxDim);

    for (ptrdiff_t i = 0; i < maxDim; ++i) {
      const element_type i_val = element_type (i) + 1.0;      
      for (ptrdiff_t j = 0; j < maxDim; ++j) {
        const element_type j_val = element_type (j) + 1.0;
        const element_type A_ij = i_val + element_type (maxDim) * j_val;

        A(i,j) = A_ij;
      }
    }
  }
}
