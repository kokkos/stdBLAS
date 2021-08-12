namespace KokkosKernelsSTD {

namespace {
template<class Accessor>
double get_scaling_factor(Accessor) { return 1.0; }

template<class Accessor, class S>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor,S> a) { return a.scale_factor(); }
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

} // namespace KokkosKernelsSTD
