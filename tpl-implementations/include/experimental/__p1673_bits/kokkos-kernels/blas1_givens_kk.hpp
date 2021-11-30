
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GIVENS_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_GIVENS_HPP_

#include <cmath>
#include <complex>

namespace KokkosKernelsSTD {

template<std::floating_point Real>
void givens_rotation_setup(const Real f,
                           const Real g,
                           Real& cs,
                           Real& sn,
                           Real& r)
{}

template<std::floating_point Real>
void givens_rotation_setup(const std::complex<Real>& f,
                           const std::complex<Real>& g,
                           Real& cs,
                           std::complex<Real>& sn,
                           std::complex<Real>& r)
{}

// c and s are std::floating_point
template<class ElementType1,
         std::experimental::extents<>::size_type ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
         std::experimental::extents<>::size_type ext2,
         class Layout2,
         class Accessor2,
         std::floating_point Real>
void givens_rotation_apply(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType1, std::experimental::extents<ext1>, Layout1, Accessor1> x,
  std::experimental::mdspan<ElementType2, std::experimental::extents<ext2>, Layout2, Accessor2> y,
  const Real c,
  const Real s)
{}

// c is std::floating_point
// s is std::complex<std::floating_point>
template<class ElementType1,
         std::experimental::extents<>::size_type ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
         std::experimental::extents<>::size_type ext2,
         class Layout2,
         class Accessor2,
         std::floating_point Real>
void givens_rotation_apply(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType1, std::experimental::extents<ext1>, Layout1, Accessor1> x,
  std::experimental::mdspan<ElementType2, std::experimental::extents<ext2>, Layout2, Accessor2> y,
  const Real c,
  const std::complex<Real> s)
{}

} // end namespace KokkosKernelsSTD
#endif
