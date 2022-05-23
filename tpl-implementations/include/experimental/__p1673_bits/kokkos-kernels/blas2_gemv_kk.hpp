
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMV_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMV_HPP_

#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"

namespace KokkosKernelsSTD {

namespace gemv_impl{
template<class Accessor, class T>
T get_scaling_factor(Accessor /*a*/, T /*v*/) {
  return T(1);
}

template<class Accessor, class S, class T>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor,S> a,
      T /*v*/)
{
  return T(a.scale_factor());
}

} //end gemv_impl namespace


//
// overwriting gemv: y = Ax
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y>
void matrix_vector_product(kokkos_exec<ExeSpace> /*kexe*/,
			   std::experimental::mdspan<
			     ElementType_A,
			     std::experimental::extents<numRows_A, numCols_A>,
			     Layout_A,
			     std::experimental::default_accessor<ElementType_A>
			   > A,
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
			   > y)
{

  // preconditions
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }

  // mandates
  Impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  Impl::static_extent_match(A.static_extent(0), y.static_extent(0));

  Impl::signal_kokkos_impl_called("overwriting_matrix_vector_product");

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  // make alpha and beta consistent with types in KokkosBlas::gemv
  const auto alpha = static_cast<typename decltype(A_view)::value_type>(1);
  const auto beta  = static_cast<typename decltype(y_view)::value_type>(0);
  KokkosBlas::gemv("N", alpha, A_view, x_view, beta, y_view);
}

//
// updating gemv: z = y + Ax
//
// for now, specialize for default_accessor
// https://github.com/kokkos/stdBLAS/issues/122
//
template<class ExeSpace,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
	 class Layout_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ext_z,
         class Layout_z>
void matrix_vector_product(kokkos_exec<ExeSpace> /*kexe*/,
			   std::experimental::mdspan<
			     ElementType_A,
			     std::experimental::extents<numRows_A, numCols_A>,
			     Layout_A,
			     std::experimental::default_accessor<ElementType_A>
			   > A,
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
			   std::experimental::mdspan<
			     ElementType_z,
			     std::experimental::extents<ext_z>,
			     Layout_z,
			     std::experimental::default_accessor<ElementType_z>
			   > z)
{

  // preconditions
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }
  if ( A.extent(0) != z.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != z.extent(0) ");
  }

  // mandates
  Impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  Impl::static_extent_match(A.static_extent(0), y.static_extent(0));
  Impl::static_extent_match(y.static_extent(0), z.static_extent(0));

  Impl::signal_kokkos_impl_called("updating_matrix_vector_product");

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  auto ex = ExeSpace();
  Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, z_view.extent(0)),
		       KOKKOS_LAMBDA (const std::size_t i)
		       {
			 typename decltype(z_view)::value_type z_i = {};
			 for (std::size_t j=0; j<A_view.extent(1); ++j){
			   z_i += y_view(i) + A_view(i,j) * x_view(j);
			 }
			 z_view(i) = z_i;
		       });

  //fence message when using latest kokkos:
  ex.fence();
  // ex.fence("KokkosStdBlas::gemv: fence after operation");
}

//
// the following specialization is here to avoid breaking
// the example code started by CTrott in this file:
//    stdBLAS/examples/kokkos-based/matrix_vector_product_kokkos.cpp
// but we need to change this later after we properly handle accessors
//
template<//class ExecSpace,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
	 class Accessor_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ext_z,
         class Layout_z>
std::enable_if_t<
  !std::is_same_v< Accessor_y, std::experimental::default_accessor<ElementType_y>>
  >
matrix_vector_product(kokkos_exec<>,
			   std::experimental::mdspan<
			    ElementType_A,
			    std::experimental::extents<numRows_A, numCols_A>,
			    Layout_A, Accessor_A
			   > A,
			   std::experimental::mdspan<
			    ElementType_x,
			    std::experimental::extents<ext_x>,
			    Layout_x,
			    Accessor_x
			   > x,
			   std::experimental::mdspan<
			    ElementType_y,
			    std::experimental::extents<ext_y>,
			    Layout_y,
			    Accessor_y
			   > y,
			   std::experimental::mdspan<
			    ElementType_z,
			    std::experimental::extents<ext_z>,
			    Layout_z,
			    std::experimental::default_accessor<ElementType_z>
			   > z)
{
  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  // here we need to compute: z = y + Ax
  // where A,x,y can have non-trivial accessors.
  // Since kokkos-kernels gemv only supports
  //     v=beta*v + alpha*A*v1
  // we cannot use it directly, so for now we
  // break the impl into two calls

  using Av_t = typename decltype(A)::value_type;
  using xv_t = typename decltype(x)::value_type;
  using yv_t = typename decltype(y)::value_type;
  using zv_t = typename decltype(z)::value_type;

  // z = alpha1*y, we know from constraint y has non default accessor
  auto alpha1 = y.accessor().scale_factor();
  constexpr zv_t zero(0);
  KokkosBlas::axpby(alpha1, y_view, zero, z_view);

  // z = z + alpha2*A*x
  using alpha_t = decltype(std::declval<Av_t>() * std::declval<xv_t>());
  auto alpha2 = gemv_impl::get_scaling_factor(A.accessor(), alpha_t{}) *
    gemv_impl::get_scaling_factor(x.accessor(), alpha_t{});

  constexpr zv_t one(1);
  KokkosBlas::gemv("N", alpha2, A_view, x_view, one, z_view);
}


} // namespace KokkosKernelsSTD
#endif
