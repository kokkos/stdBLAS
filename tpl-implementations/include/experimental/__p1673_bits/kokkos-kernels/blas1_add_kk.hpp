
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_

namespace KokkosKernelsSTD {

namespace addimpl{
template<class T, class Accessor>
T get_scaling_factor(Accessor /* a */) {
  return static_cast<T>(1);
}

template<class T, class Accessor, class S>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor, S> a) {
  return a.scale_factor();
}
} // end namespace addimpl

template<class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ... ext_z,
         class Layout_z,
         class Accessor_z>
  requires (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z))
void add(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x ...>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y ...>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z ...>, Layout_z, Accessor_z> z)
{
  static_assert(z.rank() <= 2);

  auto x_view = Impl::mdspan_to_view(x);
  using x_view_type = decltype(x_view);

  auto y_view = Impl::mdspan_to_view(y);
  using y_view_type = decltype(y_view);

  auto z_view = Impl::mdspan_to_view(z);
  using z_view_type = decltype(z_view);

  // we only need scaling factors for x,y because add overwrite z
  const auto alpha = addimpl::get_scaling_factor<ElementType_x>(x.accessor());
  const auto beta  = addimpl::get_scaling_factor<ElementType_y>(y.accessor());

  KokkosBlas::update(alpha, x_view, beta, y_view, 0, z_view);
}

}
#endif
