
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_COPY_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_COPY_HPP_

namespace KokkosKernelsSTD {

template<class ExecSpace,
	 class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y>
  requires (sizeof...(ext_x) == sizeof...(ext_y))
void copy(kokkos_exec<ExecSpace>,
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
	  > y)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "copy: kokkos impl\n";
#endif
  static_assert(x.rank() <= 2);

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  if constexpr(x.rank()==1){
    Kokkos::parallel_for(Kokkos::RangePolicy(ExecSpace(), 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t & i){
			   y_view(i) = x_view(i);
			 });
  }

  else{
    Kokkos::parallel_for(Kokkos::RangePolicy(ExecSpace(), 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t & i)
			 {
			   for (std::size_t j=0; j<x_view.extent(1); ++j){
			     y_view(i,j) = x_view(i,j);
			   }
			 });
  }
}

} // end namespace KokkosKernelsSTD
#endif
