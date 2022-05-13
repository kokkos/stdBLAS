
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_RANK2K_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_RANK2K_UPDATE_HPP_

/* #include <complex>
#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"
#include "parallel_matrix.hpp" */

namespace KokkosKernelsSTD {

// Rank-2k update of a symmetric matrix
// performs BLAS xSYR2K: C += A*trans(B) + B*trans(A)

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    class Triangle,
  /* requires */ (Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
               or Impl::is_layout_blas_packed_v<Layout_C>))
void symmetric_matrix_rank_2k_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(B.rank() == 2);
  static_assert(C.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_C, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(B.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(C.static_extent(0), C.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(1), B.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: A.extent(0) != C.extent(0)");
  }
  if ( B.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: B.extent(0) != C.extent(0)");
  }
  if ( C.extent(0) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: C.extent(0) != C.extent(1)");
  }
  if ( A.extent(1) != B.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: A.extent(1) != C.extent(1)");
  }

  Impl::signal_kokkos_impl_called("symmetric_matrix_rank_2k_update");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  const auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  using size_type = std::experimental::extents<>::size_type;
  const auto A_ext1 = A.extent(1); // = B.extent(1)
  Impl::ParallelMatrixVisitor v(ExecSpace(), C_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      auto &c = C_view(i, j);
      for (size_type k = 0; k < A_ext1; ++k) {
        c += A_view(i, k) * B_view(j, k) + B_view(i, k) * A_view(j, k);
      }
    });
}

// Rank-2k update of a Hermitian matrix
// performs BLAS xHER2K: C += A*trans(conj(B)) + B*trans(conj(A))

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    class Triangle,
  /* requires */ (Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
               or Impl::is_layout_blas_packed_v<Layout_C>))
void hermitian_matrix_rank_2k_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(B.rank() == 2);
  static_assert(C.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_C, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(B.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(C.static_extent(0), C.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(1), B.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: A.extent(0) != C.extent(0)");
  }
  if ( B.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: B.extent(0) != C.extent(0)");
  }
  if ( C.extent(0) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: C.extent(0) != C.extent(1)");
  }
  if ( A.extent(1) != B.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_2k_update: A.extent(1) != C.extent(1)");
  }

  Impl::signal_kokkos_impl_called("hermitian_matrix_rank_2k_update");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  const auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  using size_type = std::experimental::extents<>::size_type;
  using std::experimental::linalg::impl::conj_if_needed;
  const auto A_ext1 = A.extent(1); // = B.extent(1)
  Impl::ParallelMatrixVisitor v(ExecSpace(), C_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      auto &c = C_view(i, j);
      for (size_type k = 0; k < A_ext1; ++k) {
        c += A_view(i, k) * conj_if_needed(B_view(j, k))
           + B_view(i, k) * conj_if_needed(A_view(j, k));
      }
    });
}

} // namespace KokkosKernelsSTD
#endif
