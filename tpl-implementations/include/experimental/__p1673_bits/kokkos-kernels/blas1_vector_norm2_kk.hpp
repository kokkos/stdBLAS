
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_NORM2_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_NORM2_HPP_

namespace KokkosKernelsSTD {

template<class ExecSpace,
         class ElementType,
	 std::experimental::extents<>::size_type ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_norm2(kokkos_exec<ExecSpace>,
		    std::experimental::mdspan<ElementType, std::experimental::extents<ext0>, Layout, Accessor> v,
		    Scalar init)
{
  return init + KokkosBlas::nrm2(Impl::mdspan_to_view(v));
}

}
#endif
