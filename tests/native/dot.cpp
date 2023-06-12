#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>

// FIXME (mfh 2022/06/17) Temporarily disable calling the BLAS,
// to get PR testing workflow running with mdspan tag.
#if 0
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
#endif // 0

namespace {
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using std::experimental::linalg::dot;
  using std::experimental::linalg::dotc;

  TEST(BLAS1_dot, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    scalar_t expectedDotResult{};
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k = scalar_t(k) + 1.0;
      const scalar_t y_k = scalar_t(k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
    }

    const auto dotResult = dot(x, y, scalar_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResult)>, scalar_t> );
    EXPECT_EQ( dotResult, expectedDotResult );

    const auto dotResultPlusOne = dot(x, y, scalar_t{} + scalar_t(1.0));
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResultPlusOne)>, scalar_t> );
    EXPECT_EQ( dotResultPlusOne, expectedDotResult + scalar_t(1.0) );

    const auto dotResultTwoArg = dot(x, y);
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResultTwoArg)>, scalar_t> );
    EXPECT_EQ( dotResultTwoArg, expectedDotResult );

// FIXME (mfh 2022/06/17) Temporarily disable calling the BLAS,
// to get PR testing workflow running with mdspan tag.
#if 0
#ifdef LINALG_ENABLE_BLAS
    const scalar_t blasResult =
      ddot_wrapper(x.extent(0), x.data(), 1, y.data(), 1);
    EXPECT_EQ( dotResult, blasResult );
#endif // LINALG_ENABLE_BLAS
#endif // 0

    const scalar_t conjDotResult = dotc(x, y, scalar_t{});
    EXPECT_EQ( conjDotResult, expectedDotResult );

    // scalar_t dotResultPar {};
    // See note above.
    //std::experimental::dot (std::execution::par, x, y, dotResultPar);

    // This is noncomforming, but I need some way to test the executor overloads.
    //using fake_executor_t = int;
    //dot (fake_executor_t (), x, y, dotResultPar);
    //EXPECT_EQ( dotResultPar, expectedDotResult );
  }

  TEST(BLAS1_dot, mdspan_complex_double_test1)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    scalar_t expectedDotResult{};
    scalar_t expectedConjDotResult{};
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      const scalar_t y_k(real_t(k) + 2.0, real_t(k) + 2.0);
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
      using std::conj;
      expectedConjDotResult += conj(x_k) * (y_k);
    }

    const auto dotResult = dot(x, y, scalar_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResult)>, scalar_t> );
    EXPECT_EQ( dotResult, expectedDotResult );

    const auto conjDotResult = dotc(x, y, scalar_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(conjDotResult)>, scalar_t> );
    EXPECT_EQ( conjDotResult, expectedConjDotResult );

    const auto dotResultTwoArg = dot(x, y);
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResultTwoArg)>, scalar_t> );
    EXPECT_EQ( dotResultTwoArg, expectedDotResult );

    const auto conjDotResultTwoArg = dotc(x, y);
    static_assert( std::is_same_v<std::remove_const_t<decltype(conjDotResultTwoArg)>, scalar_t> );
    EXPECT_EQ( conjDotResultTwoArg, expectedConjDotResult );

    //scalar_t dotResultPar {};
    // See note above.
    //std::experimental::dot (std::execution::par, x, y, dotResultPar);

    // This is noncomforming, but I need some way to test the executor overloads.
    //using fake_executor_t = int;
    //dot (fake_executor_t (), x, y, dotResultPar);
    //EXPECT_EQ( dotResultPar, expectedDotResult );
  }

  TEST(BLAS1_dot, mdspan_complex_double_test2)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<std::size_t, dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = std::size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    scalar_t expectedDotResult{};
    scalar_t expectedConjDotResult{};
    for (std::size_t k = 0; k < vectorSize; ++k)
    {

      scalar_t x_k = {};
      scalar_t y_k = {};
      if (k % 2 == 0){
	x_k = scalar_t(real_t(k) + 1.0, real_t(k) + 1.0);
	y_k = scalar_t(real_t(k) + 2.0, real_t(k) + 2.0);
      }
      else{
	x_k = scalar_t(real_t(k) - 1.0, real_t(k) + 1.0);
	y_k = scalar_t(real_t(k) + 2.0, real_t(k) - 2.0);
      }
      x(k) = x_k;
      y(k) = y_k;
      expectedDotResult += x_k * y_k;
      using std::conj;
      expectedConjDotResult += conj(x_k) * (y_k);
    }

    const auto dotResult = dot(x, y, scalar_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResult)>, scalar_t> );
    EXPECT_EQ( dotResult, expectedDotResult );

    const auto conjDotResult = dotc(x, y, scalar_t{});
    static_assert( std::is_same_v<std::remove_const_t<decltype(conjDotResult)>, scalar_t> );
    EXPECT_EQ( conjDotResult, expectedConjDotResult );

    const auto dotResultTwoArg = dot(x, y);
    static_assert( std::is_same_v<std::remove_const_t<decltype(dotResultTwoArg)>, scalar_t> );
    EXPECT_EQ( dotResultTwoArg, expectedDotResult );

    const auto conjDotResultTwoArg = dotc(x, y);
    static_assert( std::is_same_v<std::remove_const_t<decltype(conjDotResultTwoArg)>, scalar_t> );
    EXPECT_EQ( conjDotResultTwoArg, expectedConjDotResult );
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
