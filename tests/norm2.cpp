#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

//#include <execution>
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
  using std::experimental::basic_mdspan;
  using std::experimental::vector_norm2;

  TEST(BLAS1_norm2, mdspan_double)
  {
    using std::abs;
    using std::sqrt;
    using mag_t = double;
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    constexpr mag_t tol =
      mag_t(vectorSize) * std::numeric_limits<mag_t>::epsilon();

    constexpr ptrdiff_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    mag_t expectedNormResultSquared {};
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t(k) + scalar_t(1.0);
      x(k) = x_k;
      expectedNormResultSquared += x_k * x_k;
    }

    mag_t normResult {};
    vector_norm2(x, normResult);
    const mag_t expectedNormResult = sqrt(expectedNormResultSquared);
    std::cout << "normResult: " << normResult
              << ", expectedNormResult: " << expectedNormResult << std::endl;
    EXPECT_TRUE( abs(normResult - expectedNormResult) <= tol );

#ifdef LINALG_ENABLE_BLAS
    const mag_t blasResult =
      dnrm2_wrapper(int(vectorSize), x.data(), 1);
    EXPECT_TRUE( abs(blasResult - expectedNormResult) <= tol );
#endif // LINALG_ENABLE_BLAS
  }

  TEST(BLAS1_norm2, mdspan_complex_double)
  {
    using real_t = double;
    using mag_t = real_t;
    using scalar_t = std::complex<real_t>;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize(5);
    // Complex numbers use more arithmetic than their real analogs.
    constexpr mag_t tol = 4.0 * mag_t(vectorSize) *
      std::numeric_limits<mag_t>::epsilon();

    constexpr ptrdiff_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    mag_t expectedNormResultSquared {};
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 3.0, -real_t(k) - 1.0);
      x(k) = x_k;
      expectedNormResultSquared += abs(x_k) * abs(x_k);
    }

    mag_t normResult {};
    vector_norm2(x, normResult);
    const mag_t expectedNormResult = sqrt(expectedNormResultSquared);
    EXPECT_TRUE( abs(normResult - expectedNormResult) <= tol );
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
