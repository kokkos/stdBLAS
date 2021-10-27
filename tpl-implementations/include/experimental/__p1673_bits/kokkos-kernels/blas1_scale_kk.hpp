
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_

namespace KokkosKernelsSTD {
template<class ExecSpace,
         class Scalar,
         class ElementType,
         std::experimental::extents<>::size_type ... ext,
         class Layout,
         class Accessor>
void scale(kokkos_exec<ExecSpace>, const Scalar alpha,
           std::experimental::mdspan<ElementType, std::experimental::extents<ext ...>, Layout, Accessor> x)
{
    KokkosBlas::scal(Impl::mdspan_to_view(x),alpha,Impl::mdspan_to_view(x));
}
}

#endif
