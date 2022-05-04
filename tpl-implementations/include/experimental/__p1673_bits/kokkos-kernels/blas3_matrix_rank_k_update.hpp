
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_RANK_k_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_RANK_k_UPDATE_HPP_

#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"
#include "parallel_matrix.hpp"

namespace KokkosKernelsSTD {

// Rank-k update of a symmetric matrix with scaling factor alpha
// performs BLAS xSYRK: C += alpha * A * trans(A)

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ScaleFactorType,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    class Triangle,
  /* requires */ (Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
               or Impl::is_layout_blas_packed_v<Layout_C>))
void symmetric_matrix_rank_k_update(kokkos_exec<ExecSpace> &&exec,
  ScaleFactorType alpha,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(C.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_C, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(C.static_extent(0), C.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_k_update: A.extent(0) != C.extent(0)");
  }
  if ( C.extent(0) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_k_update: C.extent(0) != C.extent(1)");
  }

  Impl::signal_kokkos_impl_called("symmetric_matrix_rank_k_update");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  auto C_view = Impl::mdspan_to_view(C);

  using size_type = std::experimental::extents<>::size_type;
  const auto A_ext1 = A.extent(1);
  Impl::ParallelMatrixVisitor v(ExecSpace(), C_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      auto &c = C_view(i, j);
      for (size_type k = 0; k < A_ext1; ++k) {
        c += alpha * A_view(i, k) * A_view(j, k);
      }
    });
}

// Rank-k update of a hermitian matrix with scaling factor alpha
// performs BLAS xHERK: C += alpha * A * trans(A)

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ScaleFactorType,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    class Triangle,
  /* requires */ (Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
               or Impl::is_layout_blas_packed_v<Layout_C>))
void hermitian_matrix_rank_k_update(kokkos_exec<ExecSpace> &&exec,
  ScaleFactorType alpha,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(C.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_C, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(C.static_extent(0), C.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_k_update: A.extent(0) != C.extent(0)");
  }
  if ( C.extent(0) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_k_update: C.extent(0) != C.extent(1)");
  }

  Impl::signal_kokkos_impl_called("hermitian_matrix_rank_k_update");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  auto C_view = Impl::mdspan_to_view(C);

  using size_type = std::experimental::extents<>::size_type;
  using std::experimental::linalg::impl::conj_if_needed;
  const auto A_ext1 = A.extent(1);
  Impl::ParallelMatrixVisitor v(ExecSpace(), C_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      auto &c = C_view(i, j);
      for (size_type k = 0; k < A_ext1; ++k) {
        c += alpha * A_view(i, k) * conj_if_needed(A_view(j, k));
      }
    });
}

} // namespace KokkosKernelsSTD
#endif
