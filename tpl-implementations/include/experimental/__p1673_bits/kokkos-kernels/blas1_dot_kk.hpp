
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_DOT_HPP_

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

namespace KokkosKernelsSTD {

template<class ExeSpace,
	 class ElementType_x,
	 std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
	 std::experimental::extents<>::size_type ext_y,
         class Layout_y,
	 class Scalar>
Scalar dot(kokkos_exec<ExeSpace> /*kexe*/,
	   std::experimental::mdspan<
	     ElementType_x,
	     std::experimental::extents<ext_x>,
	     Layout_x,
	     std::experimental::default_accessor<ElementType_x>
	   > x,
	   std::experimental::mdspan<
	     ElementType_y,
	     std::experimental::extents<ext_y>,
	     Layout_y,
	     std::experimental::default_accessor<ElementType_y>
	   > y,
	   Scalar init)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "dot: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  // this overload is for the default_accessor (see the args above).
  // Therefore, we cannot use KokkosBlas::dot here because
  // it would automatically conjugate x for the complex case.
  // Since here we have the default accessors, we DO NOT want to conjugate x,
  // we just need to compute sum(x*y), even for the complex case.
  Scalar result = {};
  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExeSpace(), 0, x_view.extent(0)),
			  KOKKOS_LAMBDA (const std::size_t i, Scalar & update) {
			    update += x_view(i)*y_view(i);
			  }, result);
  return result + init;
}

template<class ExeSpace,
	 class ElementType_x,
	 std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
	 std::experimental::extents<>::size_type ext_y,
         class Layout_y,
	 class Scalar>
Scalar dot(kokkos_exec<ExeSpace>,
	   std::experimental::mdspan<
	    ElementType_x,
	    std::experimental::extents<ext_x>,
	    Layout_x,
	    std::experimental::linalg::accessor_conjugate<
	     std::experimental::default_accessor<ElementType_x>, ElementType_x
	    >
	   > x,
	   std::experimental::mdspan<
	     ElementType_y,
	     std::experimental::extents<ext_y>,
	     Layout_y,
	     std::experimental::default_accessor<ElementType_y>
	   > y,
	   Scalar init)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "dot: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  // this overload is for x with conjugated (with nested default) accessor
  // so can call KokkosBlas::dot because it automatically conjugates x
  // and it is what we want.
  return KokkosBlas::dot(x_view, y_view) + init;
}

}
#endif
