
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_

#include <KokkosBlas1_iamax.hpp>

namespace KokkosKernelsSTD {

template<class ExecSpace,
         class ElementType,
         std::experimental::extents<>::size_type ext0,
         class Layout,
         class Accessor>
auto idx_abs_max(kokkos_exec<ExecSpace>,
		 std::experimental::mdspan<ElementType, std::experimental::extents<ext0>, Layout, Accessor> v)
{
  // note that -1 here, this is related to:
  // https://github.com/kokkos/stdBLAS/issues/114

  return KokkosBlas::iamax(Impl::mdspan_to_view(v))-1;
}

}
#endif
