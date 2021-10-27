
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_

namespace KokkosKernelsSTD {

template<
  class ElementType1,
  std::experimental::extents<>::size_type ext1,
  class Layout1,
  class Accessor1,
  class ElementType2,
  std::experimental::extents<>::size_type ext2,
  class Layout2,
  class Accessor2,
  class Scalar
  >
Scalar dot(kokkos_exec<>,
	   std::experimental::mdspan<ElementType1, std::experimental::extents<ext1>, Layout1, Accessor1> v1,
	   std::experimental::mdspan<ElementType2, std::experimental::extents<ext2>, Layout2, Accessor2> v2,
	   Scalar init)
{
  const auto result = KokkosBlas::dot(Impl::mdspan_to_view(v1), Impl::mdspan_to_view(v2));
  return init + result;
}

}
#endif
