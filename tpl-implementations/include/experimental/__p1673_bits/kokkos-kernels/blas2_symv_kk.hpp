
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SYMV_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_SYMV_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace symv_impl{
template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  return extent1 == std::experimental::dynamic_extent ||
         extent2 == std::experimental::dynamic_extent ||
         extent1 == extent2;
}
} // end namespace symv_impl

//
// overwriting symmetric gemv: y = Ax
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Triangle,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y>
requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void symmetric_matrix_vector_product(kokkos_exec<ExeSpace> /*kexe*/,
				     std::experimental::mdspan<
				       ElementType_A,
				       std::experimental::extents<numRows_A, numCols_A>,
				       Layout_A,
				       std::experimental::default_accessor<ElementType_A>
				     > A,
				     Triangle /* tr */,
				     std::experimental::mdspan<ElementType_x,
				       std::experimental::extents<ext_x>,
				       Layout_x,
				       std::experimental::default_accessor<ElementType_x>
				     > x,
				     std::experimental::mdspan<ElementType_y,
				       std::experimental::extents<ext_y>,
				       Layout_y,
				       std::experimental::default_accessor<ElementType_y>
				     > y)
{

  // preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != A.extent(1) ");
  }
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }

  symv_impl::static_extent_match(A.static_extent(0), A.static_extent(1));
  symv_impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  symv_impl::static_extent_match(A.static_extent(0), x.static_extent(0));

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  auto ex = ExeSpace();

  if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
  {

    Impl::signal_kokkos_impl_called("overwriting_symmetric_matrix_vector_product_upper");

    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t i)
			 {

			   typename decltype(y_view)::value_type lsum  = {};
			   for (std::size_t j = i; j < A_view.extent(1); ++j) {
			     lsum += A_view(i,j) * x_view(j);
			   }
			   for (std::size_t j = 0; j < i; ++j) {
			     lsum += A_view(j,i) * x_view(j);
			   }

			   y_view(i) = lsum;
			 });

    //fence message when using latest kokkos:
    ex.fence();
    // ex.fence("KokkosStdBlas::overwriting_symv_upper: fence after operation");
  }
  else{

    Impl::signal_kokkos_impl_called("overwriting_symmetric_matrix_vector_product_lower");

    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t i)
			 {

			   typename decltype(y_view)::value_type lsum = {};
			   for (std::size_t j = 0; j <= i; ++j) {
			     lsum += A_view(i,j) * x_view(j);
			   }
			   for (std::size_t j = i+1; j < A.extent(1); ++j) {
			     lsum += A_view(j,i) * x_view(j);
			   }

			   y_view(i) = lsum;
			 });

    //fence message when using latest kokkos:
    ex.fence();
    // ex.fence("KokkosStdBlas::overwriting_symv_lower: fence after operation");
  }

}


//
// updating symmetric gemv: z = y + Ax
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
	 class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Triangle,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
	 class Layout_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ext_z,
         class Layout_z>
requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void symmetric_matrix_vector_product(kokkos_exec<ExeSpace> /*kexe*/,
				     std::experimental::mdspan<
				       ElementType_A,
				       std::experimental::extents<numRows_A, numCols_A>,
				       Layout_A,
				       std::experimental::default_accessor<ElementType_A>
				     > A,
				     Triangle /* tr */,
				     std::experimental::mdspan<ElementType_x,
				       std::experimental::extents<ext_x>,
				       Layout_x,
				       std::experimental::default_accessor<ElementType_x>
				     > x,
				     std::experimental::mdspan<ElementType_y,
				       std::experimental::extents<ext_y>,
				       Layout_y,
				       std::experimental::default_accessor<ElementType_y>
				     > y,
				     std::experimental::mdspan<ElementType_z,
				       std::experimental::extents<ext_z>,
				       Layout_z,
				       std::experimental::default_accessor<ElementType_z>
				     > z)
{

  // preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != A.extent(1) ");
  }
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }
  if ( A.extent(0) != z.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != z.extent(0) ");
  }

  symv_impl::static_extent_match(A.static_extent(0), A.static_extent(1));
  symv_impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  symv_impl::static_extent_match(A.static_extent(0), x.static_extent(0));
  symv_impl::static_extent_match(y.static_extent(0), z.static_extent(0));

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  auto ex = ExeSpace();

  if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
  {

    Impl::signal_kokkos_impl_called("updating_symmetric_matrix_vector_product_upper");

    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t i){
			   typename decltype(y_view)::value_type lsum = {};
			   for (std::size_t j = i; j < A_view.extent(1); ++j) {
			     lsum += A_view(i,j) * x_view(j);
			   }
			   for (std::size_t j = 0; j < i; ++j) {
			     lsum += A_view(j,i) * x_view(j);
			   }

			   z_view(i) = y_view(i) + lsum;
			 });

    //fence message when using latest kokkos:
    ex.fence();
    // ex.fence("KokkosStdBlas::updating_symv_upper: fence after operation");
  }
  else{

    Impl::signal_kokkos_impl_called("updating_symmetric_matrix_vector_product_lower");

    Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			 KOKKOS_LAMBDA (const std::size_t i){
			   typename decltype(y_view)::value_type lsum = {};
			   for (std::size_t j = 0; j <= i; ++j) {
			     lsum += A_view(i,j) * x_view(j);
			   }
			   for (std::size_t j = i+1; j < A.extent(1); ++j) {
			     lsum += A_view(j,i) * x_view(j);
			   }

			   z_view(i) = y_view(i) + lsum;
			 });

    //fence message when using latest kokkos:
    ex.fence();
    // ex.fence("KokkosStdBlas::updating_symv_lower: fence after operation");
  }

}


} // namespace KokkosKernelsSTD
#endif
