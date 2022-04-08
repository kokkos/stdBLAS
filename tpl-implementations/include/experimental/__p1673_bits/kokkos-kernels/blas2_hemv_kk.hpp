
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_HERMITIAN_MATVEC_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_HERMITIAN_MATVEC_HPP_

namespace KokkosKernelsSTD {

namespace hermitianmatvecimpl{

template<class T> struct is_complex : std::false_type{};
template<> struct is_complex<std::complex<float>> : std::true_type{};
template<> struct is_complex<std::complex<double>> : std::true_type{};
template<> struct is_complex<std::complex<long double>> : std::true_type{};

template<class T> inline constexpr bool is_complex_v = is_complex<T>::value;

} // end namespace hermitianmatvecimpl

//
// Overwriting hermitian matrix-vector product: y = Ax
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
void hermitian_matrix_vector_product(kokkos_exec<ExeSpace> kexe,
				      std::experimental::mdspan<
					ElementType_A,
					std::experimental::extents<numRows_A, numCols_A>,
					Layout_A,
				        std::experimental::default_accessor<ElementType_A>
				      > A,
				      Triangle tr,
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

  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(0) != A.extent(1) ");
  }
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(0) != y.extent(0) ");
  }

  // for non-complex ELementType_A, A can be treated as symmetric
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1673r6.html#hermitian-matrix-vector-product-linalgalgsblas2hemv
  //
  // If inout_matrix_t::element_type is complex<RA> for some RA, then ....
  // Otherwise, the functions will assume that A[j,i] equals A[i,j].
  //
  if constexpr (hermitianmatvecimpl::is_complex_v<ElementType_A> == false){
    std::experimental::linalg::symmetric_matrix_vector_product(kexe, A, tr, x, y);
  }
  else
    {

      auto A_view = Impl::mdspan_to_view(A);
      auto x_view = Impl::mdspan_to_view(x);
      auto y_view = Impl::mdspan_to_view(y);

      auto ex = ExeSpace();

      if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
      {

	// this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
	std::cout << "overwriting_hermitian_matrix_vector_product_upper: kokkos impl\n";
#endif

	Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			     KOKKOS_LAMBDA (const std::size_t & i)
			     {

			       using lsum_type = decltype( A_view(0,0) * x_view(0) );
			       lsum_type lsum = {};

			       for (std::size_t j = i; j < A_view.extent(1); ++j) {
				 lsum += A_view(i,j) * x_view(j);
			       }
			       for (std::size_t j = 0; j < i; ++j) {
				 lsum += Kokkos::conj(A_view(j,i)) * x_view(j);
			       }

			       y_view(i) = lsum;
			     });

	ex.fence();
      }
      else{

	// this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
	std::cout << "overwriting_hermitian_matrix_vector_product_lower: kokkos impl\n";
#endif

        Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			     KOKKOS_LAMBDA (const std::size_t & i)
			     {

			       using lsum_type = decltype( A_view(0,0) * x_view(0) );
			       lsum_type lsum = {};

			       for (std::size_t j = 0; j <= i; ++j) {
				 lsum += A_view(i,j) * x_view(j);
			       }
			       for (std::size_t j = i+1; j < A.extent(1); ++j) {
				 lsum += Kokkos::conj(A_view(j,i)) * x_view(j);
			       }

			       y_view(i) = lsum;
			     });
        ex.fence();
      }
    }
}


//
// Updating hermitian matrix-vector product: z = y + Ax
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
void hermitian_matrix_vector_product(kokkos_exec<ExeSpace> kexe,
				      std::experimental::mdspan<
					ElementType_A,
					std::experimental::extents<numRows_A, numCols_A>,
					Layout_A,
				        std::experimental::default_accessor<ElementType_A>
				      > A,
				      Triangle tr,
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

  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);
  static_assert(z.rank() == 1);

  // preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(0) != A.extent(1) ");
  }
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_vector_product: A.extent(0) != y.extent(0) ");
  }
  if ( A.extent(0) != z.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_vector_product: A.extent(0) != z.extent(0) ");
  }

  // for non-complex ELementType_A, A can be treated as symmetric
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1673r6.html#hermitian-matrix-vector-product-linalgalgsblas2hemv
  //
  // If inout_matrix_t::element_type is complex<RA> for some RA, then ....
  // Otherwise, the functions will assume that A[j,i] equals A[i,j].
  //
  if constexpr (hermitianmatvecimpl::is_complex_v<ElementType_A> == false){
    std::experimental::linalg::symmetric_matrix_vector_product(kexe, A, tr, x, y, z);
  }
  else
    {

      auto A_view = Impl::mdspan_to_view(A);
      auto x_view = Impl::mdspan_to_view(x);
      auto y_view = Impl::mdspan_to_view(y);
      auto z_view = Impl::mdspan_to_view(z);

      auto ex = ExeSpace();

      if constexpr (std::is_same_v<Triangle, std::experimental::linalg::upper_triangle_t>)
      {

	// this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
	std::cout << "updating_hermitian_matrix_vector_product_upper: kokkos impl\n";
#endif

	Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			     KOKKOS_LAMBDA (const std::size_t & i)
			     {

			       using lsum_type = decltype( A_view(0,0) * x_view(0) );
			       lsum_type lsum = {};

			       for (std::size_t j = i; j < A_view.extent(1); ++j) {
				 lsum += A_view(i,j) * x_view(j);
			       }
			       for (std::size_t j = 0; j < i; ++j) {
				 lsum += Kokkos::conj(A_view(j,i)) * x_view(j);
			       }

			       z_view(i) = y_view(i) + lsum;
			     });

	ex.fence();
      }
      else{

	// this print is detected in the tests
#if defined KOKKOS_STDBLAS_ENABLE_TESTS
	std::cout << "updating_hermitian_matrix_vector_product_lower: kokkos impl\n";
#endif

        Kokkos::parallel_for(Kokkos::RangePolicy(ex, 0, A_view.extent(0)),
			     KOKKOS_LAMBDA (const std::size_t & i)
			     {

			       using lsum_type = decltype( A_view(0,0) * x_view(0) );
			       lsum_type lsum = {};

			       for (std::size_t j = 0; j <= i; ++j) {
				 lsum += A_view(i,j) * x_view(j);
			       }
			       for (std::size_t j = i+1; j < A.extent(1); ++j) {
				 lsum += Kokkos::conj(A_view(j,i)) * x_view(j);
			       }

			       z_view(i) = y_view(i) + lsum;
			     });
        ex.fence();
      }
    }
}


} // namespace KokkosKernelsSTD
#endif
