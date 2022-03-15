
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_ABS_SUM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_ABS_SUM_HPP_

namespace KokkosKernelsSTD {

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

template<class ExeSpace,
         class ElementType,
	 std::experimental::extents<>::size_type ext0,
         class Layout,
         class Scalar>
Scalar vector_abs_sum(kokkos_exec<ExeSpace> /*kexe*/,
		      std::experimental::mdspan<
		      ElementType,
		      std::experimental::extents<ext0>,
		      Layout,
		      std::experimental::default_accessor<ElementType>
		      > x,
		      Scalar init)
{
#if defined LINALG_ENABLE_TESTS
  std::cout << "vector_abs_sum: kokkos impl\n";
#endif

  auto x_view = Impl::mdspan_to_view(x);
  using arithm_traits = Kokkos::Details::ArithTraits<ElementType>;

  Scalar result = {};
  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExeSpace(), 0, x_view.extent(0)),
			  KOKKOS_LAMBDA (const std::size_t i, Scalar & update) {
			    update += arithm_traits::abs(x_view(i));
			  }, result);
  // fence not needed because reducing into result

  return result + init;
}

}
#endif
