
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_

#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"

namespace KokkosKernelsSTD {

namespace add_impl{

template <class F, class T, T... Is>
void repeat_impl(F&& f, std::integer_sequence<T, Is...>){
  ( f(std::integral_constant<T, Is>{}), ... );
}

template <int N, class F>
void repeat(F&& f){
  repeat_impl(f, std::make_integer_sequence<int, N>{});
}

} // namespace add_impl

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
  static_assert(z.rank() <= 2);

  // P1673 preconditions
  add_impl::repeat<x.rank()>
    ([=](int r){
      if ( x.extent(r) != y.extent(r) ){
	throw std::runtime_error("KokkosBlas: add: x.extent(r) != y.extent(r) for r="
				 + std::to_string(r));
      }
      if ( y.extent(r) != z.extent(r) ){
	throw std::runtime_error("KokkosBlas: add: y.extent(r) != z.extent(r) for r = "
				 + std::to_string(r));
      }
    });

  // P1673 mandates
  add_impl::repeat<x.rank()>
    ([=](int r){
      Impl::static_extent_match(x.static_extent(r), z.static_extent(r));
      Impl::static_extent_match(y.static_extent(r), z.static_extent(r));
      Impl::static_extent_match(x.static_extent(r), y.static_extent(r));
    });

  Impl::signal_kokkos_impl_called("add");

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  const auto alpha = static_cast<typename decltype(x_view)::non_const_value_type>(1);
  const auto beta  = static_cast<typename decltype(y_view)::non_const_value_type>(1);
  const auto zero  = static_cast<typename decltype(z_view)::non_const_value_type>(0);

  KokkosBlas::update(alpha, x_view, beta, y_view, zero, z_view);
}

}
#endif
