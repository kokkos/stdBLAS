
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_ONE_NORM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_ONE_NORM_HPP_

namespace KokkosKernelsSTD {

template<
    class ElementType,
    std::experimental::extents<>::size_type numRows,
    std::experimental::extents<>::size_type numCols,
    class Layout,
    class Accessor,
    class Scalar>
Scalar matrix_one_norm(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType, std::experimental::extents<numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  return {};
}

} // end namespace KokkosKernelsSTD
#endif
