
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_NORM2_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_NORM2_HPP_

namespace KokkosKernelsSTD {

template<class ExeSpace,
         class ElementType,
	 std::experimental::extents<>::size_type ext,
         class Layout,
         class Scalar>
Scalar vector_norm2(kokkos_exec<ExeSpace>,
		    std::experimental::mdspan<
		    ElementType,
		    std::experimental::extents<ext>,
		    Layout,
		    std::experimental::default_accessor<ElementType>> x,
		    Scalar init)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "vector_norm2: kokkos impl\n";
#endif

  // for the code in stBLAS/examples/kokkos-based,
  // when using float, the nrm2 does not work, giving:
  // Kokkos result = -36893488147419103232.000000
  //return KokkosBlas::nrm2(Impl::mdspan_to_view(x)) + init;

  // the following works
  using IPT = Kokkos::Details::InnerProductSpaceTraits<ElementType>;
  auto x_view = Impl::mdspan_to_view(x);
  Scalar result = {};
  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExeSpace(), 0, x_view.extent(0)),
			  KOKKOS_LAMBDA (const std::size_t i, Scalar & update) {
			    const typename IPT::mag_type tmp = IPT::norm(x_view(i));
			    update += tmp*tmp;
			  }, result);
  return Kokkos::Details::ArithTraits<Scalar>::sqrt(result + init);

}

}
#endif
