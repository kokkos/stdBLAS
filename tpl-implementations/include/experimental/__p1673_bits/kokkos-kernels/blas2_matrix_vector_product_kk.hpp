
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATVEC_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATVEC_HPP_

namespace KokkosKernelsSTD {

namespace {
template<class Accessor>
double get_scaling_factor(Accessor) { return 1.0; }

template<class Accessor, class S>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor,S> a) {
  return a.scale_factor();
}
} //end anon namespace

template<//class ExecSpace,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
void matrix_vector_product(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  auto alpha = get_scaling_factor(A.accessor())  *
               get_scaling_factor(x.accessor());
  KokkosBlas::gemv("N", alpha, Impl::mdspan_to_view(A), Impl::mdspan_to_view(x), 0.0, Impl::mdspan_to_view(y));
}

template<//class ExecSpace,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
void matrix_vector_product(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  // for now, this is split into two calls until
  // we implement the full kernel in kokkos-kernels

  // note that we need to account for scaled accessors via
  // coefficients or things will not work correctly.
  // mdspan_to_view does not account for the accessor.
  // so we need to be careful with this.
  // In fact, x(0) can be != x_view(0)
  // for example for scaled accessor.

  // z = alpha*y
  auto alpha1 = get_scaling_factor(y.accessor());
  KokkosBlas::axpby(alpha1, y_view, 0.0, z_view);

  // z = y + A * x
  auto alpha2 = get_scaling_factor(A.accessor()) * get_scaling_factor(x.accessor());
  KokkosBlas::gemv("N", alpha2, Impl::mdspan_to_view(A), x_view, 1.0, z_view);
}

} // namespace KokkosKernelsSTD
#endif
