
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

  auto A_view = Impl::mdspan_to_view(A);

  // here we use an impl similar to the scaled_sum_of_squares
  // but we do not call that directly because it would requre
  // flattening the matrix whereas this impl works for any layout

  using arithm_traits = Kokkos::Details::ArithTraits<ElementType>;

  std::experimental::linalg::sum_of_squares_result<Scalar> ssqr;
  ssqr.scaling_factor = {};
  ssqr.scaled_sum_of_squares = {};

  Kokkos::Max<Scalar> max_reducer(ssqr.scaling_factor);
  Kokkos::parallel_reduce( Kokkos::RangePolicy(ExeSpace(), 0, A_view.extent(0)*A_view.extent(1)),
			   KOKKOS_LAMBDA (const std::size_t k, Scalar & lmax){
			     const auto i = k / A_view.extent(1);
			     const auto j = k % A_view.extent(1);
			     const auto val = arithm_traits::abs(A_view(i,j));
			     max_reducer.join(lmax, val);
			   },
			   max_reducer);
  // no fence needed since reducing into scalar

  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExeSpace(), 0, A_view.extent(0)*A_view.extent(1)),
			  KOKKOS_LAMBDA (const std::size_t k, Scalar & update){
			    const auto i = k / A_view.extent(1);
			    const auto j = k % A_view.extent(1);
			    const auto tmp = arithm_traits::abs(A_view(i,j))/ssqr.scaling_factor;
			    update += tmp*tmp;
			  }, ssqr.scaled_sum_of_squares);
  // no fence needed since reducing into scalar

  return std::sqrt(init + ssqr.scaling_factor * ssqr.scaling_factor * ssqr.scaled_sum_of_squares);
}

} // end namespace KokkosKernelsSTD
#endif
