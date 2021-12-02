
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_

namespace KokkosKernelsSTD {

namespace dotimpl{

template<class T, class Accessor>
T get_scaling_factor(Accessor /* a */) {
  return static_cast<T>(1);
}

template<class T, class Accessor, class S>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor, S> a) {
  return a.scale_factor();
}
} // end namespace dotimpl

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
	   std::experimental::mdspan<ElementType1, std::experimental::extents<ext1>, Layout1, Accessor1> x,
	   std::experimental::mdspan<ElementType2, std::experimental::extents<ext2>, Layout2, Accessor2> y,
	   Scalar init)
{
  // we cannot use KokkosBlas::dot because it does not allow scaling factors
  // signature is: dot (const VectorX& X, const VectorY& Y);
  // and we cannot just modify x,y

  auto x_view = Impl::mdspan_to_view(x);
  using x_view_type = decltype(x_view);

  auto y_view = Impl::mdspan_to_view(y);
  using y_view_type = decltype(y_view);

  // doing this for now, but we should do something here like we did
  // for Kokkos::reduce when we don't have a neutral value
  Scalar sum = {};
  Kokkos::parallel_reduce("stdBLAS::KokkosKernelsSTD_dot", x_view.extent(0),
			  KOKKOS_LAMBDA (const std::size_t i, Scalar & update) {
			    update += x_view(i) * y_view(i);
			  }, sum);


  // we only need scaling factors for x,y because add overwrite z
  const auto alpha = addimpl::get_scaling_factor<ElementType1>(x.accessor());
  const auto beta  = addimpl::get_scaling_factor<ElementType2>(y.accessor());

  return alpha * beta * sum + init;
}

}
#endif
