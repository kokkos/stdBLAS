
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_

#include <KokkosBlas1_iamax.hpp>
#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

template<class ExeSpace,
         class ElementType,
         std::experimental::extents<>::size_type ext0,
         class Layout>
std::experimental::extents<>::size_type
idx_abs_max(kokkos_exec<ExeSpace> /*kexe*/,
	    std::experimental::mdspan<
	    ElementType,
	    std::experimental::extents<ext0>,
	    Layout,
	    std::experimental::default_accessor<ElementType>> v)
{
  Impl::signal_kokkos_impl_called("idx_abs_max");

  auto v_view = Impl::mdspan_to_view(v);

  // note that -1 here, this is related to:
  // https://github.com/kokkos/stdBLAS/issues/114
  return KokkosBlas::iamax(v_view) - 1;
}

}
#endif
