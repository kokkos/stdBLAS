
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_UPDATING_GEMM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_UPDATING_GEMM_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace upd_gemm_impl{
template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  return extent1 == std::experimental::dynamic_extent ||
         extent2 == std::experimental::dynamic_extent ||
         extent1 == extent2;
}
} //end upd_gemm_impl namespace

// to complete after we finish the overwriting

// //
// // updating gemm: C = E + AB
// //
// MDSPAN_TEMPLATE_REQUIRES(
//          class ExeSpace,
// 	 class ElementType_A,
//          std::experimental::extents<>::size_type numRows_A,
//          std::experimental::extents<>::size_type numCols_A,
//          class Layout_A,
// 	 class Accessor_A,
//          class ElementType_B,
//          std::experimental::extents<>::size_type numRows_B,
//          std::experimental::extents<>::size_type numCols_B,
//          class Layout_B,
// 	 class Accessor_B,
//          class ElementType_E,
//          std::experimental::extents<>::size_type numRows_E,
//          std::experimental::extents<>::size_type numCols_E,
//          class Layout_E,
// 	 class Accessor_E,
//          class ElementType_C,
//          std::experimental::extents<>::size_type numRows_C,
//          std::experimental::extents<>::size_type numCols_C,
//          class Layout_C,
// 	 class Accessor_C,
// 	 /* requires */
//          (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique() &&
// 	  Layout_B::template mapping<std::experimental::extents<numRows_B, numCols_B>>::is_always_unique() &&
// 	  Layout_E::template mapping<std::experimental::extents<numRows_E, numCols_E>>::is_always_unique() &&
// 	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique()
// 	  ))
// void matrix_product(
//   kokkos_exec<ExeSpace> kexe,
//   std::experimental::mdspan<
//     ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A
//   > A,
//   std::experimental::mdspan<
//     ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B
//   > B,
//   std::experimental::mdspan<
//     ElementType_E, std::experimental::extents<numRows_E, numCols_E>, Layout_E, Accessor_E
//   > E,
//   std::experimental::mdspan<
//     ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C
//   > C)
// {}

} // namespace KokkosKernelsSTD
#endif
