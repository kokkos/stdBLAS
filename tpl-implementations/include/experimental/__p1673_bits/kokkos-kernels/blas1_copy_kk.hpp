
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_COPY_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_COPY_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

template<class ExeSpace,
	 class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y>
void copy(kokkos_exec<ExeSpace> /*kexe*/,
	  std::experimental::mdspan<
	    ElementType_x,
	    std::experimental::extents<ext_x ...>,
	    Layout_x,
	    std::experimental::default_accessor<ElementType_x>
	  > x,
	  std::experimental::mdspan<
	    ElementType_y,
	    std::experimental::extents<ext_y ...>,
	    Layout_y,
	    std::experimental::default_accessor<ElementType_y>
	  > y)
requires ((sizeof...(ext_x) == sizeof...(ext_y)) && (x.rank() <=2))
{
  Impl::signal_kokkos_impl_called("copy");

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto ex = ExeSpace();

  if constexpr(std::is_same_v<typename decltype(x_view)::array_layout, typename decltype(y_view)::array_layout>) {
    Kokkos::deep_copy(ex, y, x);
  } else {

    if constexpr(x.rank()==1){
      Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, x_view.extent(0)),
			   KOKKOS_LAMBDA (const std::size_t i){
			     y_view(i) = x_view(i);
			   });
    }

    else{
      Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, x_view.extent(0)),
			   KOKKOS_LAMBDA (const std::size_t i){
			     for (std::size_t j=0; j<x_view.extent(1); ++j){
			       y_view(i,j) = x_view(i,j);
			     }
			   });
    }
  }

  // need to fence even for deep_copy case since passing
  // ex to it makes deep_copy potentially non-blocking
  // https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3Adeep_copy
  ex.fence();
  //fence message when using latest kokkos:
  // ex.fence("KokkosStdBlas::copy: fence after operation");
}

} // end namespace KokkosKernelsSTD
#endif
