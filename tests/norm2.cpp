#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

//#include <execution>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"

#ifdef LINALG_ENABLE_BLAS
extern "C" double dnrm2_(const int* pN,
                         const double* X,
                         const int* pINCX);

double dnrm2_wrapper(const int N, const double* X, const int INCX)
{
  return dnrm2_(&N, X, &INCX);
}
#endif // LINALG_ENABLE_BLAS

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::vector_norm2;

  TEST(BLAS1_norm2, mdspan_zero)
  {
    // This test ensures that vectors with no entries have a norm of exactly 0

    using mag_t = double;
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(0);
    std::vector<scalar_t> storage(vectorSize);
    vector_t x(storage.data(), vectorSize);

    // Testing for absolute equality
    const auto normResult = vector_norm2(x, mag_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResult)>, mag_t> );
    const mag_t expectedNormResult{};
    EXPECT_EQ( expectedNormResult, normResult );

    // Make sure that init always gets added to the result.
    const mag_t normResultPlusOne = vector_norm2(x, mag_t(1.0));
    EXPECT_EQ( expectedNormResult + mag_t(1.0), normResultPlusOne );

    // Test 'auto' overload.
    const auto normResultAuto = vector_norm2(x);
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResultAuto)>, mag_t> );
    EXPECT_EQ( expectedNormResult, normResultAuto );
  }

  TEST(BLAS1_norm2, mdspan_one)
  {
    // This test ensures that vectors with one entry have a norm of exactly the magnitude of the only element

    using std::abs;
    using real_t = double;
    using mag_t = real_t;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(1);
    std::vector<scalar_t> storage(vectorSize);
    vector_t x(storage.data(), vectorSize);

    x[0] = -3;

    // Testing for absolute equality
    const auto normResult = vector_norm2(x, mag_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResult)>, mag_t> );
    const mag_t expectedNormResult = abs( x[0] );
    EXPECT_EQ( expectedNormResult, normResult );

    // Make sure that init always gets added to the result.
    const mag_t normResultPlusOne = vector_norm2(x, mag_t(1.0));
    EXPECT_EQ( expectedNormResult + mag_t(1.0), normResultPlusOne );

    // Test 'auto' overload.
    const auto normResultAuto = vector_norm2(x);
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResultAuto)>, mag_t> );
    EXPECT_EQ( expectedNormResult, normResultAuto );
  }

  TEST(BLAS1_norm2, mdspan_double)
  {
    using std::abs;
    using std::sqrt;
    using mag_t = double;
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr mag_t tol =
      mag_t(vectorSize) * std::numeric_limits<mag_t>::epsilon();

    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    // Set elements in descending order so the scaling triggers
    mag_t expectedNormResultSquared {};
    for (std::size_t k = vectorSize; k > 1; --k) {
      const scalar_t x_k = scalar_t(k);
      x(k-1) = x_k;
      expectedNormResultSquared += x_k * x_k;
    }

    const auto normResult = vector_norm2(x, mag_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResult)>, mag_t> );
    const mag_t expectedNormResult = sqrt(expectedNormResultSquared);
    EXPECT_NEAR( expectedNormResult, normResult, tol );

    // Test 'auto' overload.
    const auto normResultAuto = vector_norm2(x);
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResultAuto)>, mag_t> );
    EXPECT_NEAR( expectedNormResult, normResultAuto, tol );

#ifdef LINALG_ENABLE_BLAS
    const mag_t blasResult =
      dnrm2_wrapper(int(vectorSize), x.data(), 1);
    EXPECT_NEAR( expectedNormResult, blasResult, tol );
#endif // LINALG_ENABLE_BLAS
  }

  TEST(BLAS1_norm2, mdspan_complex_double)
  {
    using real_t = double;
    using mag_t = real_t;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    // Complex numbers use more arithmetic than their real analogs.
    constexpr mag_t tol = 4.0 * mag_t(vectorSize) *
      std::numeric_limits<mag_t>::epsilon();

    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    mag_t expectedNormResultSquared {};
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 3.0, -real_t(k) - 1.0);
      x(k) = x_k;
      expectedNormResultSquared += abs(x_k) * abs(x_k);
    }

    const auto normResult = vector_norm2(x, mag_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResult)>, mag_t> );
    const mag_t expectedNormResult = sqrt(expectedNormResultSquared);
    EXPECT_NEAR( expectedNormResult, normResult, tol );

    // Test 'auto' overload.
    const auto normResultAuto = vector_norm2(x);
    static_assert( std::is_same_v<std::remove_const_t<decltype(normResultAuto)>, mag_t> );
    EXPECT_NEAR( expectedNormResult, normResultAuto, tol );
  }
}
