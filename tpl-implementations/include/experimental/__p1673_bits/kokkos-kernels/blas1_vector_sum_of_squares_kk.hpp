
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_SUM_OF_SQUARES_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_SUM_OF_SQUARES_HPP_

namespace KokkosKernelsSTD {

template<class ExecSpace,
         class ElementType,
         std::experimental::extents<>::size_type ext0,
         class Layout,
         class Accessor,
         class Scalar>
std::experimental::linalg::sum_of_squares_result<Scalar>
vector_sum_of_squares(kokkos_exec<ExecSpace>,
		      std::experimental::mdspan<ElementType, std::experimental::extents<ext0>, Layout, Accessor> v,
		      std::experimental::linalg::sum_of_squares_result<Scalar> init)
{
  std::cout << "vector_sum_of_squares: kkernels impl missing !!!\n";
  return init;
}

}
#endif
