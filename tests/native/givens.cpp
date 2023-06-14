
#define MDSPAN_USE_PAREN_OPERATOR 1

#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <limits>
#include <vector>

namespace {
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using std::experimental::linalg::givens_rotation_setup;
  using std::experimental::linalg::givens_rotation_apply;

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

  TEST(givens_rotation_apply, double)
  {
    using real_t = double;
    using scalar_t = real_t;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0);
      const scalar_t y_k(real_t(k) + 5.0);
      x(k) = x_k;
      y(k) = y_k;
    }

    {
      const real_t c(1.0);
      const scalar_t s(0.0);

      givens_rotation_apply(x, y, c, s);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        scalar_t x_k(real_t(k) + 4.0);
        scalar_t y_k(real_t(k) + 5.0);

        const scalar_t tmp = c * x_k + s * y_k;
        y_k = c * y_k - s * x_k;
        x_k = tmp;

        EXPECT_EQ( x(k), x_k );
        EXPECT_EQ( y(k), y_k );
      }
    }
    {
      const real_t c(0.0);
      const scalar_t s(1.0);

      givens_rotation_apply(x, y, c, s);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        scalar_t x_k(real_t(k) + 4.0);
        scalar_t y_k(real_t(k) + 5.0);

        const scalar_t tmp = c * x_k + s * y_k;
        y_k = c * y_k - s * x_k;
        x_k = tmp;

        EXPECT_EQ( x(k), x_k );
        EXPECT_EQ( y(k), y_k );
      }
    }
  }

  TEST(givens_rotation_apply, complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
      const scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);
      x(k) = x_k;
      y(k) = y_k;
    }

    using std::conj;

    {
      const real_t c(1.0);
      const scalar_t s(0.0, 0.0);

      givens_rotation_apply(x, y, c, s);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);

        const scalar_t tmp = c * x_k + s * y_k;
        y_k = c * y_k - conj(s) * x_k;
        x_k = tmp;

        EXPECT_EQ( x(k), x_k );
        EXPECT_EQ( y(k), y_k );
      }
    }
    {
      const real_t c(0.0);
      const scalar_t s(1.0, 0.0);

      givens_rotation_apply(x, y, c, s);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        scalar_t y_k(real_t(k) + 5.0, -real_t(k) - 2.0);

        const scalar_t tmp = c * x_k + s * y_k;
        y_k = c * y_k - conj(s) * x_k;
        x_k = tmp;

        EXPECT_EQ( x(k), x_k );
        EXPECT_EQ( y(k), y_k );
      }
    }
  }
}
