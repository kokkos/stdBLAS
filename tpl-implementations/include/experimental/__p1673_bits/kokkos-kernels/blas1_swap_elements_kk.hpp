
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SWAP_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SWAP_HPP_

#include <utility>

namespace KokkosKernelsSTD {

// this is here but until we can use kokkos 3.6 which has swap avail
template <class T>
KOKKOS_INLINE_FUNCTION void _my_tmp_swap(T& a, T& b) noexcept {
  static_assert(
      std::is_move_assignable<T>::value && std::is_move_constructible<T>::value,
      "KokkosKernelsSTD::swap arguments must be move assignable and move constructible");

  T tmp = std::move(a);
  a     = std::move(b);
  b     = std::move(tmp);
}

template<class ExeSpace,
	 class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y>
  requires (sizeof...(ext_x) == sizeof...(ext_y))
void swap_elements(kokkos_exec<ExeSpace> /*kexe*/,
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
  static_assert(x.rank() <= 2);

#if defined LINALG_ENABLE_TESTS
  std::cout << "swap_elements: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  if constexpr(x.rank()==1){
    Kokkos::parallel_for(Kokkos::RangePolicy(ExeSpace(), 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t & i){
			   _my_tmp_swap(x_view(i), y_view(i));
			 });
  }

  else{
    Kokkos::parallel_for(Kokkos::RangePolicy(ExeSpace(), 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t & i)
			 {
			   for (std::size_t j=0; j<x_view.extent(1); ++j){
			     _my_tmp_swap(x_view(i,j), y_view(i,j));
			   }
			 });
  }
}

} // end namespace KokkosKernelsSTD
#endif
