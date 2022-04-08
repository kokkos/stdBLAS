
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_

namespace KokkosKernelsSTD {

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

template<class ExeSpace,
   class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ... ext_z,
         class Layout_z>
  requires (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z))
void add(kokkos_exec<ExeSpace>,
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
   > y,
   std::experimental::mdspan<
     ElementType_z,
     std::experimental::extents<ext_z ...>,
     Layout_z,
     std::experimental::default_accessor<ElementType_z>
   > z)
{
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
  std::cout << "add: kokkos impl\n";
#endif

  static_assert(z.rank() <= 2);

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  const auto alpha = static_cast<ElementType_x>(1);
  const auto beta  = static_cast<ElementType_y>(1);
  const auto zero  = static_cast<ElementType_z>(0);

  KokkosBlas::update(alpha, x_view, beta, y_view, zero, z_view);
}

}
#endif
