#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

#include <complex>

namespace KokkosKernelsSTD {

namespace Impl {

// manages parallel execution of independent action
// called like action(i, j) for each matrix element A(i, j)
template <typename ExecSpace, typename MatrixType, typename ActionType>
void for_each_matrix_element(ExecSpace &&exec, MatrixType &A, ActionType action) {
  const auto num_rows = A.extent(0);
  const auto num_cols = A.extent(1);

  const auto max_val = std::numeric_limits<decltype(num_rows * num_cols)>::max();
  if (num_rows < max_val / num_cols) { // parallel elements
    Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, num_rows * num_cols),
      KOKKOS_LAMBDA(const auto ij) {
        const auto i = ij / num_cols;
        const auto j = ij - (i * num_cols); // = ij % num_cols
        action(i, j);
      });
  } else {
    // parallelize over single dimension due to index int overflow
    if (num_rows > num_cols) { // parallel rows
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, num_rows),
        KOKKOS_LAMBDA(const auto i) {
          using idx_type = std::remove_const_t<decltype(i)>;
          for (idx_type j = 0; j < num_cols; ++j) {
            action(i, j);
          }
        });
    } else { // parallel columns
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, num_cols),
        KOKKOS_LAMBDA(const auto j) {
          using idx_type = std::remove_const_t<decltype(j)>;
          for (idx_type i = 0; i < num_rows; ++i) {
            action(i, j);
          }
        });
      }
  }
  exec.fence();
}

// This version of conj_if_needed() also handles Kokkos::complex<T>
template <class T>
KOKKOS_INLINE_FUNCTION
T conj_if_needed(const T &value)
{
  return value;
};

template <class T>
KOKKOS_INLINE_FUNCTION
auto conj_if_needed(const Kokkos::complex<T> &value)
{
  return Kokkos::conj(value);
};

template <class T>
KOKKOS_INLINE_FUNCTION
auto conj_if_needed(const std::complex<T> &value)
{
  return std::conj(value);
};

} // namespace Impl

// Performs BLAS xGER/xGERU (for real/complex types):
// A[i,j] += x[i] * y[j]
template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A>
void matrix_rank_1_update(kokkos_exec<ExecSpace> &&/* exec */,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::default_accessor<ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A)
{
  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(1) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(1) != y.extent(0)");
  }

#if defined LINALG_ENABLE_TESTS
  std::cout << "matrix_rank1_update: kokkos impl\n";
#endif

  // convert mdspans to views and wrap input with original accessors
  const auto x_view = Impl::mdspan_to_view(x);
  const auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);

  Impl::for_each_matrix_element(ExecSpace(), A_view,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * y_view(j);
    });
}

// conjugated(y) specialization dispatched by matrix_rank_1_update_c
template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A>
void matrix_rank_1_update(kokkos_exec<ExecSpace> &&/* exec */,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::linalg::accessor_conjugate<
      std::experimental::default_accessor<ElementType_y>, ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A)
{
  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(1) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(1) != y.extent(0)");
  }

#if defined LINALG_ENABLE_TESTS
  std::cout << "matrix_rank1_update: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);

  Impl::for_each_matrix_element(ExecSpace(), A_view,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      // apply conjugation explicitly (accessor is no longer on the view)
      A_view(i, j) += x_view(i) * Impl::conj_if_needed(y_view(j));
    });
}

} // namespace KokkosKernelsSTD
#endif
