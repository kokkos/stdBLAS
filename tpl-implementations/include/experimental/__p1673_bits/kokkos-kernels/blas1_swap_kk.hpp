
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_

#include <utility>

namespace KokkosKernelsSTD {

template<class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y,
         class Accessor_y>
  requires (sizeof...(ext_x) == sizeof...(ext_y))
void swap_elements(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x ...>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y ...>, Layout_y, Accessor_y> y)
{
}

} // end namespace KokkosKernelsSTD
#endif
