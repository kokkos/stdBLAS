
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SWAP_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SWAP_HPP_

#include <utility>
#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace swap_impl{

// this is here until we can use kokkos 3.6 which has swap avail
template <class T>
requires(std::is_move_assignable<T>::value && std::is_move_constructible<T>::value)
KOKKOS_INLINE_FUNCTION void _my_tmp_swap(T& a, T& b) noexcept
{
  T tmp = std::move(a);
  a     = std::move(b);
  b     = std::move(tmp);
}

template <class F, class T, T... Is>
void repeat_impl(F&& f, std::integer_sequence<T, Is...>){
  ( f(std::integral_constant<T, Is>{}), ... );
}

template <int N, class F>
void repeat(F&& f){
  repeat_impl(f, std::make_integer_sequence<int, N>{});
}

template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  return extent1 == std::experimental::dynamic_extent ||
         extent2 == std::experimental::dynamic_extent ||
         extent1 == extent2;
}

} // end namespace swap_impl

//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
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
  // matching rank already checked via requires above
  static_assert(x.rank() <= 2);

  // P1673 preconditions
  swap_impl::repeat<x.rank()>
    ([=](int r){
      if ( x.extent(r) != y.extent(r) ){
	throw std::runtime_error("KokkosBlas: swap_elements: x.extent(r) != y.extent(r) for r="
				 + std::to_string(r));
      }
    });

  // P1673 mandates
  swap_impl::repeat<x.rank()>
    ([=](int r){
      swap_impl::static_extent_match(x.static_extent(r), y.static_extent(r));
    });

  Impl::signal_kokkos_impl_called("swap_elements");

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  auto ex = ExeSpace();
  if constexpr(x.rank()==1){
    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (std::size_t i){
			   swap_impl::_my_tmp_swap(x_view(i), y_view(i));
			 });
  }

  else{
    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, x_view.extent(0)),
			 KOKKOS_LAMBDA (std::size_t i){
			   for (std::size_t j=0; j<x_view.extent(1); ++j){
			     swap_impl::_my_tmp_swap(x_view(i,j), y_view(i,j));
			   }
			 });
  }

  //fence message when using latest kokkos:
  ex.fence();
  // ex.fence("KokkosStdBlas::swap_elements: fence after operation");
}

} // end namespace KokkosKernelsSTD
#endif
