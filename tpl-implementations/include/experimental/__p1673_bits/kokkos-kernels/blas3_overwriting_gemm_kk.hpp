
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_OVERWRITING_GEMM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_OVERWRITING_GEMM_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace ov_gemm_impl{
template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  return extent1 == std::experimental::dynamic_extent ||
         extent2 == std::experimental::dynamic_extent ||
         extent1 == extent2;
}
} //end ov_gemm_impl namespace

//
// overwriting gemm: C = alpha*A*B
//
MDSPAN_TEMPLATE_REQUIRES(
         class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
	 class Accessor_A,
         class ElementType_B,
         std::experimental::extents<>::size_type numRows_B,
         std::experimental::extents<>::size_type numCols_B,
         class Layout_B,
	 class Accessor_B,
         class ElementType_C,
         std::experimental::extents<>::size_type numRows_C,
         std::experimental::extents<>::size_type numCols_C,
         class Layout_C,
	 class Accessor_C,
	 /* requires */
	 (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique() &&
	  Layout_B::template mapping<std::experimental::extents<numRows_B, numCols_B>>::is_always_unique() &&
	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique()
	  ))
void matrix_product(
  kokkos_exec<ExeSpace> /*kexe*/,
  std::experimental::mdspan<
    ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A
  > A,
  std::experimental::mdspan<
    ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B
  > B,
  std::experimental::mdspan<
    ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C
  > C)
{

  // preconditions
  if ( A.extent(1) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_AB_product: A.extent(1) != B.extent(0) ");
  }
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_AB_product: A.extent(0) != C.extent(0) ");
  }
  if ( B.extent(1) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: gemm_C_AB_product: B.extent(1) != C.extent(1) ");
  }

  // mandates
  ov_gemm_impl::static_extent_match(A.static_extent(1), B.static_extent(0));
  ov_gemm_impl::static_extent_match(A.static_extent(0), C.static_extent(0));
  ov_gemm_impl::static_extent_match(B.static_extent(1), C.static_extent(1));

  auto A_view = Impl::mdspan_to_view(A);
  auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  namespace stdexp = std::experimental;

  constexpr bool doing_C_equal_A_B =
    std::is_same_v<Accessor_A, stdexp::default_accessor<ElementType_A>> &&
    std::is_same_v<Accessor_B, stdexp::default_accessor<ElementType_B>> &&
    std::is_same_v<Accessor_C, stdexp::default_accessor<ElementType_C>>;

  if constexpr(doing_C_equal_A_B){
    Impl::signal_kokkos_impl_called("gemm_C_AB_product");

    const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
    const auto beta  = static_cast<typename decltype(C_view)::value_type>(0);
    KokkosBlas::gemm("N", "N", alpha, A_view, B_view, beta, C_view);
  }
  // todo: handle cases stemming from non-trivial accessors

  else{
    // fallback serial impl
    std::experimental::linalg::matrix_product(A,B,C);
  }
}

//
// overwriting gemm: C = alpha * A^T B
//
MDSPAN_TEMPLATE_REQUIRES(
         class ExeSpace,
	 class ElementType_AT,
         std::experimental::extents<>::size_type numRows_AT,
         std::experimental::extents<>::size_type numCols_AT,
         class NestedLayout_A,
	 class Accessor_AT,
         class ElementType_B,
         std::experimental::extents<>::size_type numRows_B,
         std::experimental::extents<>::size_type numCols_B,
         class Layout_B,
	 class Accessor_B,
         class ElementType_C,
         std::experimental::extents<>::size_type numRows_C,
         std::experimental::extents<>::size_type numCols_C,
         class Layout_C,
	 class Accessor_C,
	 /* requires */
	 (NestedLayout_A::template mapping<std::experimental::extents<numRows_AT, numCols_AT>>::is_always_unique() &&
	  Layout_B::template mapping<std::experimental::extents<numRows_B, numCols_B>>::is_always_unique() &&
	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique()
	  ))
void matrix_product(
  kokkos_exec<ExeSpace> /*kexe*/,
  std::experimental::mdspan<
    ElementType_AT, std::experimental::extents<numRows_AT, numCols_AT>,
    std::experimental::linalg::layout_transpose<NestedLayout_A>, Accessor_AT
  > AT,
  std::experimental::mdspan<
    ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B
  > B,
  std::experimental::mdspan<
    ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C
  > C)
{

  // note that AT = A^T so AT carries the tranpose effect

  // preconditions
  if ( AT.extent(1) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ATB_product: AT.extent(1) != B.extent(0) ");
  }
  if ( AT.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ATB_product: AT.extent(0) != C.extent(0) ");
  }
  if ( B.extent(1) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ATB_product: B.extent(1) != C.extent(1) ");
  }

  // mandates
  ov_gemm_impl::static_extent_match(AT.static_extent(1), B.static_extent(0));
  ov_gemm_impl::static_extent_match(AT.static_extent(0), C.static_extent(0));
  ov_gemm_impl::static_extent_match(B.static_extent(1), C.static_extent(1));

  // note that the conversion to view does NOT carry the transpose
  auto A_view = Impl::mdspan_to_view(AT);
  auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  namespace stdexp = std::experimental;

  constexpr bool doing_C_equal_AT_B =
    std::is_same_v<Accessor_AT, stdexp::default_accessor<ElementType_AT>> &&
    std::is_same_v<Accessor_B, stdexp::default_accessor<ElementType_B>> &&
    std::is_same_v<Accessor_C, stdexp::default_accessor<ElementType_C>>;

  if constexpr(doing_C_equal_AT_B){
    Impl::signal_kokkos_impl_called("gemm_C_ATB_product");

    const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
    const auto beta  = static_cast<typename decltype(C_view)::value_type>(0);
    // we need to explicitly pass "T" because A_view does NOT carry the transpose
    KokkosBlas::gemm("T", "N", alpha, A_view, B_view, beta, C_view);
  }
  // todo: handle cases stemming from non-trivial accessors

  else{
    // fallback serial impl
    std::experimental::linalg::matrix_product(AT,B,C);
  }
}


//
// overwriting gemm: C = alpha * A B^T
//
MDSPAN_TEMPLATE_REQUIRES(
         class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
	 class Accessor_A,
         class ElementType_BT,
         std::experimental::extents<>::size_type numRows_BT,
         std::experimental::extents<>::size_type numCols_BT,
         class NestedLayout_B,
	 class Accessor_BT,
         class ElementType_C,
         std::experimental::extents<>::size_type numRows_C,
         std::experimental::extents<>::size_type numCols_C,
         class Layout_C,
	 class Accessor_C,
	 /* requires */
	 (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique() &&
	  NestedLayout_B::template mapping<std::experimental::extents<numRows_BT, numCols_BT>>::is_always_unique() &&
	  Layout_C::template mapping<std::experimental::extents<numRows_C, numCols_C>>::is_always_unique()
	  ))
void matrix_product(
  kokkos_exec<ExeSpace> /*kexe*/,
  std::experimental::mdspan<
    ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A
  > A,
  std::experimental::mdspan<
    ElementType_BT, std::experimental::extents<numRows_BT, numCols_BT>,
    std::experimental::linalg::layout_transpose<NestedLayout_B>, Accessor_BT
  > BT,
  std::experimental::mdspan<
    ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C
  > C)
{

  // note that BT = B^T so BT carries the tranpose effect

  // preconditions
  if ( A.extent(1) != BT.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ABT_product: A.extent(1) != BT.extent(0) ");
  }
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ABT_product: A.extent(0) != C.extent(0) ");
  }
  if ( BT.extent(1) != C.extent(1) ){
    throw std::runtime_error("KokkosBlas: gemm_C_ABT_product: BT.extent(1) != C.extent(1) ");
  }

  // mandates
  ov_gemm_impl::static_extent_match(A.static_extent(1),  BT.static_extent(0));
  ov_gemm_impl::static_extent_match(A.static_extent(0),  C.static_extent(0));
  ov_gemm_impl::static_extent_match(BT.static_extent(1), C.static_extent(1));

  // note that the conversion to view does NOT carry the transpose
  auto A_view = Impl::mdspan_to_view(A);
  auto B_view = Impl::mdspan_to_view(BT);
  auto C_view = Impl::mdspan_to_view(C);

  namespace stdexp = std::experimental;

  constexpr bool doing_C_equal_A_BT =
    std::is_same_v<Accessor_A, stdexp::default_accessor<ElementType_A>> &&
    std::is_same_v<Accessor_BT, stdexp::default_accessor<ElementType_BT>> &&
    std::is_same_v<Accessor_C, stdexp::default_accessor<ElementType_C>>;

  if constexpr(doing_C_equal_A_BT){
    Impl::signal_kokkos_impl_called("gemm_C_ABT_product");

    const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
    const auto beta  = static_cast<typename decltype(C_view)::value_type>(0);
    // we need to explicitly pass "T" because B_view does NOT carry the transpose
    KokkosBlas::gemm("N", "T", alpha, A_view, B_view, beta, C_view);
  }
  // todo: handle cases stemming from non-trivial accessors

  else{
    // fallback serial impl
    std::experimental::linalg::matrix_product(A,BT,C);
  }
}

} // namespace KokkosKernelsSTD
#endif
