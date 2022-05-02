
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMM_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace gemm_impl{
template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  return extent1 == std::experimental::dynamic_extent ||
         extent2 == std::experimental::dynamic_extent ||
         extent1 == extent2;
}
} //end gemm_impl namespace

//
// overwriting gemm: C = AB
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class ElementType_B,
         std::experimental::extents<>::size_type numRows_B,
         std::experimental::extents<>::size_type numCols_B,
         class Layout_B,
         class ElementType_C,
         std::experimental::extents<>::size_type numRows_C,
         std::experimental::extents<>::size_type numCols_C,
         class Layout_C>
requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique() &&
	  Layout_B::template mapping<std::experimental::extents<numRows_B, numCols_B>>::is_always_unique() &&
	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique())
void matrix_product(
  kokkos_exec<ExeSpace> /*kexe*/,
  std::experimental::mdspan<
    ElementType_A, std::experimental::extents<numRows_A, numCols_A>,
    Layout_A, std::experimental::default_accessor<ElementType_A>
  > A,
  std::experimental::mdspan<
    ElementType_B, std::experimental::extents<numRows_B, numCols_B>,
    Layout_B, std::experimental::default_accessor<ElementType_B>
  > B,
  std::experimental::mdspan<
    ElementType_C, std::experimental::extents<numRows_C, numCols_C>,
    Layout_C, std::experimental::default_accessor<ElementType_C>
  > C)
{

  // preconditions
  if ( A.extent(1) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: A.extent(1) != B.extent(0) ");
  }
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: A.extent(0) != C.extent(0) ");
  }
  if ( B.extent(1) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: B.extent(1) != C.extent(1) ");
  }

  // mandates
  gemm_impl::static_extent_match(A.static_extent(1), B.static_extent(0));
  gemm_impl::static_extent_match(A.static_extent(0), C.static_extent(0));
  gemm_impl::static_extent_match(B.static_extent(1), C.static_extent(1));

  Impl::signal_kokkos_impl_called("overwriting_matrix_matrix_product");

  auto A_view = Impl::mdspan_to_view(A);
  auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
  const auto beta  = static_cast<typename decltype(C_view)::value_type>(0);
  KokkosBlas::gemm("N", "N", alpha, A_view, B_view, beta, C_view);
}

//
// updating gemm: C = E + AB
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class ElementType_B,
         std::experimental::extents<>::size_type numRows_B,
         std::experimental::extents<>::size_type numCols_B,
         class Layout_B,
         class ElementType_E,
         std::experimental::extents<>::size_type numRows_E,
         std::experimental::extents<>::size_type numCols_E,
         class Layout_E,
         class ElementType_C,
         std::experimental::extents<>::size_type numRows_C,
         std::experimental::extents<>::size_type numCols_C,
         class Layout_C>
requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique() &&
	  Layout_B::template mapping<std::experimental::extents<numRows_B, numCols_B>>::is_always_unique() &&
	  Layout_E::template mapping<std::experimental::extents<numRows_E, numCols_E>>::is_always_unique() &&
	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique())
void matrix_product(
  kokkos_exec<ExeSpace> kexe,
  std::experimental::mdspan<
    ElementType_A, std::experimental::extents<numRows_A, numCols_A>,
    Layout_A, std::experimental::default_accessor<ElementType_A>
  > A,
  std::experimental::mdspan<
    ElementType_B, std::experimental::extents<numRows_B, numCols_B>,
    Layout_B, std::experimental::default_accessor<ElementType_B>
  > B,
  std::experimental::mdspan<
    ElementType_E, std::experimental::extents<numRows_E, numCols_E>,
    Layout_E, std::experimental::default_accessor<ElementType_E>
  > E,
  std::experimental::mdspan<
    ElementType_C, std::experimental::extents<numRows_C, numCols_C>,
    Layout_C, std::experimental::default_accessor<ElementType_C>
  > C)
{

  // preconditions
  if ( C.extent(0) != E.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: C.extent(0) != E.extent(0) ");
  }
  if ( C.extent(1) != E.extent(1) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: C.extent(1) != E.extent(1) ");
  }
  if ( A.extent(1) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: A.extent(1) != B.extent(0) ");
  }
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: A.extent(0) != C.extent(0) ");
  }
  if ( B.extent(1) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: matrix_matrix_product: B.extent(1) != C.extent(1) ");
  }

  // mandates
  gemm_impl::static_extent_match(C.static_extent(0), E.static_extent(0));
  gemm_impl::static_extent_match(C.static_extent(1), E.static_extent(1));
  gemm_impl::static_extent_match(A.static_extent(1), B.static_extent(0));
  gemm_impl::static_extent_match(A.static_extent(0), C.static_extent(0));
  gemm_impl::static_extent_match(B.static_extent(1), C.static_extent(1));

  Impl::signal_kokkos_impl_called("updating_matrix_matrix_product");

  auto A_view = Impl::mdspan_to_view(A);
  auto B_view = Impl::mdspan_to_view(B);
  auto E_view = Impl::mdspan_to_view(E);
  auto C_view = Impl::mdspan_to_view(C);

  // C = E
  std::experimental::linalg::copy(kexe, E, C);
  ExeSpace().fence();

  // C = C + A*B
  const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
  const auto beta  = static_cast<typename decltype(C_view)::value_type>(1);
  KokkosBlas::gemm("N", "N", alpha, A_view, B_view, beta, C_view);
}

} // namespace KokkosKernelsSTD
#endif
