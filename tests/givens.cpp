#include <experimental/linalg>
//#include <experimental/mdspan>
#include <limits>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::givens_rotation_setup;

  TEST(givens_rotation_setup, complex_double)
  {
    using std::abs;
    using std::hypot;
    using std::sqrt;
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    constexpr scalar_t ZERO (0.0);
    constexpr scalar_t ONE (1.0);

    // safmin == min (smallest normalized positive floating-point
    // number) for IEEE 754 floating-point arithmetic only.
    constexpr real_t safmin (std::numeric_limits<real_t>::min());
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon();
    // Base of the floating-point arithmetic.
    constexpr real_t base (2.0); // slamch('B')
    constexpr real_t two (2.0);

    const real_t safmn2 = pow(base, int(log(safmin / eps) / log(base) / two));
    const real_t safmx2 = real_t(1.0) / safmn2;

    real_t c;
    scalar_t s, r;
    // b=0 case for various positive a.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = scalingFactor * ONE;
      const scalar_t b = ZERO;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_EQ( c*c + s*s, ONE );
      EXPECT_EQ( hypot(abs(a), abs(b)), r );
      EXPECT_EQ( c, ONE );
      EXPECT_EQ( s, ZERO );
      EXPECT_EQ( r, scalingFactor );
    }
    // a=0 case for various positive b.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = ZERO;
      const scalar_t b = scalingFactor * ONE;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_EQ( c*c + s*s, ONE );
      EXPECT_EQ( hypot(abs(a), abs(b)), r );
      EXPECT_EQ( c, ZERO );
      EXPECT_TRUE( s == -ONE || s == ONE );
      EXPECT_EQ( r, scalingFactor );
    }

    const real_t ONE_OVER_SQRT_TWO = 1.0 / sqrt(2.0);
    const real_t tol = 4.0 * eps;

    // a==b case for various positive a and b.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = scalingFactor * ONE_OVER_SQRT_TWO * ONE;
      const scalar_t b = scalingFactor * ONE_OVER_SQRT_TWO * ONE;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_TRUE( abs(c*c + s*s - ONE) <= tol );
      EXPECT_TRUE( abs(c - s) <= tol );
      EXPECT_TRUE( abs(r - scalingFactor) <= tol );
    }
  }

  TEST(givens_rotation_setup, double)
  {
    using std::abs;
    using std::hypot;
    using std::sqrt;
    using real_t = double;
    using scalar_t = real_t;
    constexpr scalar_t ZERO (0.0);
    constexpr scalar_t ONE (1.0);

    // safmin == min (smallest normalized positive floating-point
    // number) for IEEE 754 floating-point arithmetic only.
    constexpr real_t safmin (std::numeric_limits<real_t>::min());
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon();
    // Base of the floating-point arithmetic.
    constexpr real_t base (2.0); // slamch('B')
    constexpr real_t two (2.0);

    const real_t safmn2 = pow(base, int(log(safmin / eps) / log(base) / two));
    const real_t safmx2 = real_t(1.0) / safmn2;

    real_t c;
    scalar_t s, r;
    // b=0 case for various positive a.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = scalingFactor * ONE;
      const scalar_t b = ZERO;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_EQ( c*c + s*s, ONE );
      EXPECT_EQ( hypot(abs(a), abs(b)), r );
      EXPECT_EQ( c, ONE );
      EXPECT_EQ( s, ZERO );
      EXPECT_EQ( r, scalingFactor );
    }
    // a=0 case for various positive b.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = ZERO;
      const scalar_t b = scalingFactor * ONE;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_EQ( c*c + s*s, ONE );
      EXPECT_EQ( hypot(abs(a), abs(b)), r );
      EXPECT_EQ( c, ZERO );
      EXPECT_TRUE( s == -ONE || s == ONE );
      EXPECT_EQ( r, scalingFactor );
    }

    const real_t ONE_OVER_SQRT_TWO = 1.0 / sqrt(2.0);
    const real_t tol = 4.0 * eps;

    // a==b case for various positive a and b.
    for (real_t scalingFactor = 1.0;
         scalingFactor <= safmx2 * real_t(2.0);
         scalingFactor *= 2.0) {
      const scalar_t a = scalingFactor * ONE_OVER_SQRT_TWO * ONE;
      const scalar_t b = scalingFactor * ONE_OVER_SQRT_TWO * ONE;
      givens_rotation_setup(a, b, c, s, r);

      EXPECT_TRUE( abs(c*c + s*s - ONE) <= tol );
      EXPECT_TRUE( abs(c - s) <= tol );
      EXPECT_TRUE( abs(std::hypot(a, b) - r) <= tol );
      // It's really scalingFactor*eps, not sqrt(scalingFactor)*eps.
      ASSERT_TRUE( abs(scalingFactor - r) <= scalingFactor*eps );
    }
  }
}
