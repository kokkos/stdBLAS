
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMV_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GEMV_HPP_

namespace KokkosKernelsSTD {

namespace gemvimpl{
template<class Accessor>
double get_scaling_factor(Accessor) { return 1.0; }

template<class Accessor, class S>
auto get_scaling_factor(std::experimental::linalg::accessor_scaled<Accessor,S> a) {
  return a.scale_factor();
}
} //end anon namespace

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

  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }

  // this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
  std::cout << "overwriting_matrix_vector_product: kokkos impl\n";
#endif

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);

  // make alpha and beta consistent with types in KokkosBals::gemv
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

  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);
  static_assert(z.rank() == 1);

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

  // this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
  std::cout << "updating_matrix_vector_product: kokkos impl\n";
#endif

  auto A_view = Impl::mdspan_to_view(A);
  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto z_view = Impl::mdspan_to_view(z);

  // FRIZZI: we need to improve this or maybe we should
  // just do two ops: fist y->z, and then gemv directly
  auto ex = ExeSpace();
  Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, z_view.extent(0)),
		       KOKKOS_LAMBDA (const std::size_t & i)
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
         class Accessor_y>
std::enable_if_t<
  !std::is_same<Accessor_A, std::experimental::default_accessor<ElementType_A>>::value
  and !std::is_same<Accessor_x, std::experimental::default_accessor<ElementType_x>>::value
  and !std::is_same<Accessor_y, std::experimental::default_accessor<ElementType_y>>::value
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
		      > y)
{
  auto alpha = gemvimpl::get_scaling_factor(A.accessor())  *
               gemvimpl::get_scaling_factor(x.accessor());
  KokkosBlas::gemv("N", alpha, Impl::mdspan_to_view(A),
		   Impl::mdspan_to_view(x), 0.0,
		   Impl::mdspan_to_view(y));
}


} // namespace KokkosKernelsSTD
#endif
