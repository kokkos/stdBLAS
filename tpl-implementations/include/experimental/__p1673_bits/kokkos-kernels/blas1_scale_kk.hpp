
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

namespace KokkosKernelsSTD {

template<class ExecSpace,
         class Scalar,
         class ElementType,
         std::experimental::extents<>::size_type ... ext,
         class Layout,
         class Accessor>
void scale(kokkos_exec<ExecSpace>,
	   const Scalar alpha,
           std::experimental::mdspan<
	     ElementType,
	     std::experimental::extents<ext ...>,
	     Layout,
	     std::experimental::default_accessor<ElementType>
	   > x)
{
  auto x_view = Impl::mdspan_to_view(x);
  KokkosBlas::scal(x, alpha, x);
}

}
#endif
