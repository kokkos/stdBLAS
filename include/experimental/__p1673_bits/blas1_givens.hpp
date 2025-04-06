//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_GIVENS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_GIVENS_HPP_

#include <cmath>
#include <complex>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// For the mathematical description of setup_givens_rotation, see BLAS
// Standard, Section 2.8.3 ("Generate Transformations"), GEN_GROT.  In
// the complex case, the implementation is based on LAPACK's CLARTG:
//
// http://www.netlib.org/lapack/explore-html/d1/dfa/clartg_8f_source.html
//
// For justification, see LAPACK Working Note #148, "On computing
// Givens rotations reliably and efficiently."
//
// http://www.netlib.org/lapack/lawnspdf/lawn148.pdf
//
// If your hardware has fast floating-point "exception" handling, read
// that report to learn how to optimize for your platform.  The
// approach in that report has additional advantages over the BLAS'
// CROTG; for example, the definition for complex data is consistent
// with the definition for real data.
//
// I used the following rules to translate Fortran types and
// intrinsic functions into C++:
//
// DOUBLE PRECISION -> Real
// DOUBLE COMPLEX -> complex<Real>
//
// CDABS -> abs (complex<Real> input, Real return value)
// DABS -> abs (Real input and return value)
// DSQRT -> sqrt (Real input and return value)
// DSIGN -> copysign (Real input and return value)
// DCMPLX -> complex<Real> constructor (two Real input)
// DCONJG -> conj (complex<Real> input and return value)
// DSQRT -> sqrt (Real input and return value)
// slapy2(real(fs), aimag(fs)) -> hypot(real(fs), imag(fs))


// begin anonymous namespace
namespace {
template <class Exec, class x_t, class y_t, class c_t, class s_t, class = void>
struct is_custom_apply_givens_rotation_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class c_t, class s_t>
struct is_custom_apply_givens_rotation_avail<
  Exec, x_t, y_t, c_t, s_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(apply_givens_rotation
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<const c_t>(),
		std::declval<const s_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};
} // end anonymous namespace

MDSPAN_TEMPLATE_REQUIRES( class Real, /* requires */ ( MDSPAN_IMPL_TRAIT(std::is_floating_point, Real) ) )
void setup_givens_rotation(const Real f,
                           const Real g,
                           Real& cs,
                           Real& sn,
                           Real& r)
{
  //safmin = dlamch( 'S' )
  // safmin == min (smallest normalized positive floating-point
  // number) for IEEE 754 floating-point arithmetic only.
  constexpr Real safmin = std::numeric_limits<Real>::min();
  //eps = dlamch( 'E' )
  constexpr Real eps = std::numeric_limits<Real>::epsilon();
  // Base of the floating-point arithmetic.
  constexpr Real base = 2.0; // slamch('B')

  using std::abs;
  using std::log;
  using std::max;
  using std::pow;
  using std::sqrt;

  // Original Fortran expresssion:
  //
  //    safmn2 = dlamch( 'B' )**int( log( safmin / eps ) /
  // $            log( dlamch( 'B' ) ) / two )
  //
  // The ** (pow) operator has highest precedence.
  constexpr Real two (2.0);
  const Real safmn2 =
    pow(base, int(log(safmin / eps) / log(base) / two));
  const Real safmx2 = Real(1.0) / safmn2;

  if (g == 0.0) { // includes the case f == g == 0
    cs = 1.0;
    sn = 0.0;
    r = f;
  }
  else if (f == 0.0) { // g must be nonzero
    cs = 0.0;
    sn = 1.0;
    r = g;
  }
  else { // f and g both nonzero
    auto f1 = f;
    auto g1 = g;
    auto scale = max(abs(f1), abs(g1));

    if (scale >= safmx2) {
      // At least one of f1 and g1 is large; rescale to avoid hypot.
      int count = 0;
      do {
        // 10       CONTINUE
        count = count + 1;
        f1 = f1 * safmn2;
        g1 = g1 * safmn2;
        scale = max(abs(f1), abs(g1));
        // IF( scale.GE.safmx2 ) GO TO 10
      } while (scale >= safmx2);

      r = sqrt(f1*f1 + g1*g1);
      cs = f1 / r;
      sn = g1 / r;
      for (int i = 1; i <= count; ++i) {
        r = r * safmx2;
      }
    }
    else if (scale <= safmn2) {
      // f1 and g1 are both small; rescale to avoid hypot.
      int count = 0;
      do {
        // 30       CONTINUE
        count = count + 1;
        f1 = f1 * safmx2;
        g1 = g1 * safmx2;
        scale = max(abs(f1), abs(g1));
        // IF( scale.LE.safmn2 ) GO TO 30
      } while (scale <= safmn2);

      r = sqrt(f1*f1 + g1*g1);
      cs = f1 / r;
      sn = g1 / r;
      for (int i = 1; i <= count; ++i) {
        r = r * safmn2;
      }
    }
    else {
      // If f and g are not both too small, and neither of them is too
      // large, then we don't have to trouble ourselves with hypot,
      // since the usual formula won't commit unwarranted underflow or
      // overflow.
      r = sqrt(f1*f1 + g1*g1);
      cs = f1 / r;
      sn = g1 / r;
    }

    // abs( f ).GT.abs( g ) .AND. cs.LT.zero
    if (abs(f) > abs(g) && cs < 0.0) {
      cs = -cs;
      sn = -sn;
      r = -r;
    }
  }
}

namespace impl {
MDSPAN_TEMPLATE_REQUIRES( class Real, /* requires */ ( MDSPAN_IMPL_TRAIT(std::is_floating_point, Real) ) )
Real abs1(const std::complex<Real>& ff) {
  using std::abs;
  using std::imag;
  using std::max;
  using std::real;

  return max(abs(real(ff)), abs(imag(ff)));
}

MDSPAN_TEMPLATE_REQUIRES( class Real, /* requires */ ( MDSPAN_IMPL_TRAIT(std::is_floating_point, Real) ) )
Real abssq(const std::complex<Real>& ff) {
  using std::imag;
  using std::real;

  return real(ff)*real(ff) + imag(ff)*imag(ff);
}
}

MDSPAN_TEMPLATE_REQUIRES( class Real, /* requires */ ( MDSPAN_IMPL_TRAIT(std::is_floating_point, Real) ) )
void setup_givens_rotation(const std::complex<Real>& f,
                           const std::complex<Real>& g,
                           Real& cs,
                           std::complex<Real>& sn,
                           std::complex<Real>& r)
{
  using std::abs;
  using std::complex;
  using std::imag;
  using std::isnan;
  using std::log;
  using std::max;
  using std::pow;
  using std::real;

  const Real two = 2.0;
  const Real one = 1.0;
  const Real zero = 0.0;
  const complex<Real> czero (0.0, 0.0);

  // safmin == min (smallest normalized positive floating-point
  // number) for IEEE 754 floating-point arithmetic only.
  constexpr Real safmin = std::numeric_limits<Real>::min();
  constexpr Real eps = std::numeric_limits<Real>::epsilon();
  // Base of the floating-point arithmetic.
  constexpr Real base = 2.0; // slamch('B')
  const Real safmn2 = pow(base, int(log(safmin / eps) / log(base) / two));
  const Real safmx2 = one / safmn2;

  Real scale = max(impl::abs1(f), impl::abs1(g));
  auto fs = f;
  auto gs = g;
  int count = 0;
  if (scale >= safmx2) { // scale is large
label10:
    count = count + 1;
    fs = fs*safmn2;
    gs = gs*safmn2;
    scale = scale*safmn2;
    if (scale >= safmx2) {
      goto label10;
    }
  }
  else if (scale <= safmn2) { // scale is small
    if (g == czero || isnan(abs(g))) {
      cs = one;
      sn = czero;
      r = f;
      return;
    }
label20:
    count = count - 1;
    fs = fs * safmx2;
    gs = gs * safmx2;
    scale = scale * safmx2;
    if (scale <= safmn2) {
      goto label20;
    }
  }
  auto f2 = impl::abssq(fs);
  auto g2 = impl::abssq(gs);
  if (f2 <= max(g2, one) * safmin) {
    // This is a rare case: F is very small.
    if (f == czero) {
      cs = zero;
      r = hypot(real(g), imag(g));
      // Do complex/real division explicitly with two real divisions
      const auto d = hypot(real(gs), imag(gs));
      sn = complex<Real>(real(gs) / d, -imag(gs) / d);
      return;
    }
    auto f2s = hypot(real(fs), imag(fs));

    // G2 and G2S are accurate
    // G2 is at least SAFMIN, and G2S is at least SAFMN2

    auto g2s = sqrt(g2);

    // Error in CS from underflow in F2S is at most
    // UNFL / SAFMN2 .lt. sqrt(UNFL*EPS) .lt. EPS
    // If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN,
    // and so CS .lt. sqrt(SAFMIN)
    // If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN
    // and so CS .lt. sqrt(SAFMIN)/SAFMN2 = sqrt(EPS)
    // Therefore, CS = F2S/G2S / sqrt( 1 + (F2S/G2S)**2 ) = F2S/G2S

    cs = f2s / g2s;

    // Make sure abs(FF) = 1
    // Do complex/real division explicitly with 2 real divisions
    complex<Real> ff;
    if (impl::abs1(f) > one) {
      const auto d = hypot(real(f), imag(f));
      ff = complex<Real>(real(f) / d, imag(f) / d);
    }
    else {
      const auto dr = safmx2 * real(f);
      const auto di = safmx2 * imag(f);
      const auto d = hypot(dr, di);
      ff = complex<Real>(dr / d, di / d);
    }
    sn = ff * complex<Real>(real(gs) / g2s, -imag(gs) / g2s);
    r = cs * f + sn * g;
  }
  else {
    // This is the most common case.
    // Neither F2 nor F2/G2 are less than SAFMIN
    // F2S cannot overflow, and it is accurate

    const auto f2s = sqrt(one + g2 / f2);

    // Do the F2S(real)*FS(complex) multiply with two real multiplies

    r = complex<Real>(f2s * real(fs), f2s * imag(fs));
    cs = one / f2s;
    const auto d = f2 + g2;

    // Do complex/real division explicitly with two real divisions

    sn = complex<Real>(real(r) / d, imag(r) / d);
    sn = sn * conj(gs);
    if (count != 0) {
      if (count > 0) {
        for (int i = 1; i <= count; ++i) {
          r = r * safmx2;
        }
      }
      else {
        for (int i = 1; i >= -count; --i) {
          r = r * safmn2;
        }
      }
    }
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const Real s)
{
  static_assert(x.static_extent(0) == dynamic_extent ||
                y.static_extent(0) == dynamic_extent ||
                x.static_extent(0) == y.static_extent(0));

  using index_type = ::std::common_type_t<SizeType1, SizeType2>;
  const auto x_extent_0 = static_cast<index_type>(x.extent(0));
  for (index_type i = 0; i < x_extent_0; ++i) {
    const auto dtemp = c * x(i) + s * y(i);
    y(i) = c * y(i) - s * x(i);
    x(i) = dtemp;
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  ExecutionPolicy&& exec,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const Real s)
{
  constexpr bool use_custom = is_custom_apply_givens_rotation_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(x), decltype(y), Real, Real
    >::value;

  if constexpr (use_custom) {
    apply_givens_rotation(impl::map_execpolicy_with_check(exec), x, y, c, s);
  }
  else
  {
    apply_givens_rotation(impl::inline_exec_t(), x, y, c, s);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const Real s)
{
  apply_givens_rotation(impl::default_exec_t{}, x, y, c, s);
}


// c is std::floating_point
// s is complex<std::floating_point>
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const std::complex<Real> s)
{
  static_assert(x.static_extent(0) == dynamic_extent ||
                y.static_extent(0) == dynamic_extent ||
                x.static_extent(0) == y.static_extent(0));

  using std::conj;
  using index_type = ::std::common_type_t<SizeType1, SizeType2>;
  const auto x_extent_0 = static_cast<index_type>(x.extent(0));
  for (index_type i = 0; i < x_extent_0; ++i) {
    const auto dtemp = c * x(i) + s * y(i);
    y(i) = c * y(i) - conj(s) * x(i);
    x(i) = dtemp;
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  ExecutionPolicy&& exec,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const std::complex<Real> s)
{
  constexpr bool use_custom = is_custom_apply_givens_rotation_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(x), decltype(y), Real, std::complex<Real>
    >::value;

  if constexpr (use_custom) {
    apply_givens_rotation(impl::map_execpolicy_with_check(exec), x, y, c, s);
  }
  else {
    apply_givens_rotation(impl::inline_exec_t{}, x, y, c, s);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Real,
         /* requires */ (MDSPAN_IMPL_TRAIT(std::is_floating_point, Real))
)
void apply_givens_rotation(
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y,
  const Real c,
  const std::complex<Real> s)
{
  apply_givens_rotation(impl::default_exec_t{}, x, y, c, s);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_GIVENS_HPP_
