
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
         class Scalar,
         class ElementType,
         std::experimental::extents<>::size_type ... ext,
         class Layout>
requires (sizeof...(ext) <= 2)
void scale(kokkos_exec<ExeSpace> /*kexe*/,
	   const Scalar alpha,
           std::experimental::mdspan<
	   ElementType,
	   std::experimental::extents<ext ...>,
	   Layout,
	   std::experimental::default_accessor<ElementType>
	   > obj)
{

  Impl::signal_kokkos_impl_called("scale");
  auto obj_view = Impl::mdspan_to_view(obj);
  KokkosBlas::scal(obj_view, alpha, obj_view);
}

}
#endif
