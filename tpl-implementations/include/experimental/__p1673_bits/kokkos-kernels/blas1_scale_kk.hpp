
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SCALE_HPP_

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

namespace KokkosKernelsSTD {

template<class ExeSpace,
         class Scalar,
         class ElementType,
         std::experimental::extents<>::size_type ... ext,
         class Layout>
void scale(kokkos_exec<ExeSpace> /*kexe*/,
	   const Scalar alpha,
           std::experimental::mdspan<
	     ElementType,
	     std::experimental::extents<ext ...>,
	     Layout,
	     std::experimental::default_accessor<ElementType>
	   > x)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "scale: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  KokkosBlas::scal(x_view, alpha, x_view);
}

}
#endif
