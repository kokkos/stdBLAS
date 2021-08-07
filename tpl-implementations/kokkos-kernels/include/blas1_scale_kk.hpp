
#include<experimental/mdspan>
#include<experimental/linalg>
#include<KokkosBlas.hpp>
#include<exec_policy_wrapper_kk.hpp>
namespace KokkosKernelsSTD {
template<class ExecSpace,
         class Scalar,
         class ElementType,
         std::experimental::extents<>::size_type ... ext,
         class Layout,
         class Accessor>
void scale(kokkos_exec<ExecSpace>, const Scalar alpha,
           std::experimental::mdspan<ElementType, std::experimental::extents<ext ...>, Layout, Accessor> x)
{
//  if constexpr(x.rank()==2) {
    Kokkos::View<ElementType*> x_v(x.data(),x.extent(0));
    KokkosBlas::scal(x_v,alpha,x_v);
//  } else {
//  }
}
}

