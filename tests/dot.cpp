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
extern "C" double ddot_(const int* pN, const double* DX2,
                        const int* pINCX, const double* DY2,
                        const int* pINCY);

double ddot_wrapper (const int N, const double* DX,
                     const int INCX, const double* DY,
                     const int INCY)
{
  return ddot_ (&N, DX, &INCX, DY, &INCY);
}
#endif // LINALG_ENABLE_BLAS

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::linalg::dot;
  using std::experimental::linalg::dotc;

  TEST(BLAS1_dot, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize (5);
    constexpr ptrdiff_t storageSize = ptrdiff_t (2) * vectorSize;
    std::vector<scalar_t> storage (storageSize);

    vector_t x (storage.data (), vectorSize);
    vector_t y (storage.data () + vectorSize, vectorSize);

    scalar_t expectedDotResult {};
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t (k) + 1.0;
      const scalar_t y_k = scalar_t (k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
    }

    scalar_t dotResult {};
    dot(x, y, dotResult);
    EXPECT_EQ( dotResult, expectedDotResult );

#ifdef LINALG_ENABLE_BLAS
    const scalar_t blasResult =
      ddot_wrapper(x.extent(0), x.data(), 1, y.data(), 1);
    EXPECT_EQ( dotResult, blasResult );    
#endif // LINALG_ENABLE_BLAS
    
    scalar_t conjDotResult {};
    dotc(x, y, conjDotResult);
    EXPECT_EQ( conjDotResult, expectedDotResult );

    scalar_t dotResultPar {};
    // See note above.
    //std::experimental::dot (std::execution::par, x, y, dotResultPar);

    // This is noncomforming, but I need some way to test the executor overloads.
    using fake_executor_t = int;
    dot (fake_executor_t (), x, y, dotResultPar);
    EXPECT_EQ( dotResultPar, expectedDotResult );
  }

  TEST(BLAS1_dot, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize (5);
    constexpr ptrdiff_t storageSize = ptrdiff_t (2) * vectorSize;
    std::vector<scalar_t> storage (storageSize);

    vector_t x (storage.data (), vectorSize);
    vector_t y (storage.data () + vectorSize, vectorSize);

    scalar_t expectedDotResult {};
    scalar_t expectedConjDotResult {};
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      const scalar_t y_k(real_t(k) + 2.0, real_t(k) + 2.0);
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
      using std::conj;
      expectedConjDotResult += x_k * conj (y_k);
    }

    scalar_t dotResult {};
    dot (x, y, dotResult);
    EXPECT_EQ( dotResult, expectedDotResult );

    scalar_t conjDotResult {};
    dotc (x, y, conjDotResult);
    EXPECT_EQ( conjDotResult, expectedConjDotResult );

    scalar_t dotResultPar {};
    // See note above.
    //std::experimental::dot (std::execution::par, x, y, dotResultPar);

    // This is noncomforming, but I need some way to test the executor overloads.
    using fake_executor_t = int;
    dot (fake_executor_t (), x, y, dotResultPar);
    EXPECT_EQ( dotResultPar, expectedDotResult );
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
