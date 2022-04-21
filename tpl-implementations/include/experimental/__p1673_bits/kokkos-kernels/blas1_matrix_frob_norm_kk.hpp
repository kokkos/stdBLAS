
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_FROB_NORM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_FROB_NORM_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

template<
    class ExeSpace,
    class ElementType,
    std::experimental::extents<>::size_type numRows,
    std::experimental::extents<>::size_type numCols,
    class Layout,
    class Scalar>
Scalar matrix_frob_norm(kokkos_exec<ExeSpace> kexe,
			std::experimental::mdspan<
			ElementType,
			std::experimental::extents<numRows, numCols>,
			Layout,
			std::experimental::default_accessor<ElementType>> A,
			Scalar init)
{

  Impl::signal_kokkos_impl_called("matrix_frob_norm");

  // corner cases
  constexpr std::size_t zero = 0;
  constexpr std::size_t one  = 1;
  if (A.extent(0) == zero || A.extent(1) == zero) {
    return init;
  }
  else if(A.extent(0) == one && A.extent(1) == one) {
    using std::abs;
    return init + abs(A(0, 0));
  }

  const std::size_t totNumElements = A.extent(0)*A.extent(1);
  using flatten_t = std::experimental::mdspan<
    ElementType, std::experimental::extents<std::experimental::dynamic_extent>,
    Layout, std::experimental::default_accessor<ElementType>>;

  flatten_t Aflat(A.data(), totNumElements);
  std::experimental::linalg::sum_of_squares_result<Scalar> initSsq;
  initSsq.scaling_factor = Scalar{};
  initSsq.scaled_sum_of_squares = Scalar{};
  const auto res = KokkosKernelsSTD::vector_sum_of_squares(kexe, Aflat, initSsq);

  return std::sqrt(init + res.scaling_factor * res.scaling_factor * res.scaled_sum_of_squares);
}

} // end namespace KokkosKernelsSTD
#endif
