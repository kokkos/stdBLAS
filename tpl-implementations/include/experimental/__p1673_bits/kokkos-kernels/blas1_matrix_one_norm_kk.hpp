
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_ONE_NORM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_ONE_NORM_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

template<
    class ExeSpace,
    class ElementType,
    std::experimental::extents<>::size_type numRows,
    std::experimental::extents<>::size_type numCols,
    class Layout,
    class Scalar>
Scalar matrix_one_norm(kokkos_exec<ExeSpace> /*kexe*/,
			std::experimental::mdspan<
			ElementType,
			std::experimental::extents<numRows, numCols>,
			Layout,
			std::experimental::default_accessor<ElementType>> A,
			Scalar init)
{

  Impl::signal_kokkos_impl_called("matrix_one_norm");

  if (A.extent(1) == 0){
    return init;
  }

  auto A_view = Impl::mdspan_to_view(A);

  Scalar result = {};
  Kokkos::Max<Scalar> reducer(result);
  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExeSpace(), 0, A_view.extent(1)),
			  KOKKOS_LAMBDA (const std::size_t j, Scalar & update)
			  {
			    using ats = Kokkos::Details::ArithTraits<ElementType>;
			    Scalar mysum = ats::abs(A_view(0,j));
			    for (std::size_t i=1; i<A_view.extent(0); ++i){
			      mysum += ats::abs(A_view(i,j));
			    }
			    reducer.join(update, mysum);
			  }, reducer);

  // fence not needed because reducing into result

  return init + result;
}

} // end namespace KokkosKernelsSTD
#endif
