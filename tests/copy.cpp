#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

#include <complex>
//#include <execution>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::linalg_copy;

  template<class Real>
  struct MakeVectorValues {
    static std::pair<Real, Real> make (const size_t k) {
      const Real x_k = Real(k) + Real(1.0);
      const Real y_k = Real(k) + Real(2.0);
      return {x_k, y_k};
    }
  };

  template<class Real>
  struct MakeVectorValues<std::complex<Real>> {
    static std::pair<std::complex<Real>, std::complex<Real>>
    make (const size_t k) {
      const std::complex<Real> x_k(Real(k) + 4.0, -Real(k) - 1.0);
      const std::complex<Real> y_k(Real(k) + 5.0, -Real(k) - 2.0);
      return {x_k, y_k};
    }
  };

  template<class Scalar>
  std::pair<Scalar, Scalar> makeVectorValues(const size_t k) {
    return MakeVectorValues<Scalar>::make(k);
  }

  TEST(BLAS1_copy_vector, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr size_t vectorSize(5);
    constexpr size_t storageSize = size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (size_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      x(k) = vals.first;
      y(k) = vals.second;
    }

    linalg_copy(x, y);
    for (size_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), vals.first );
      EXPECT_EQ( y(k), vals.first ); // check the output
    }
  }

  TEST(BLAS1_copy_vector, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = basic_mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr size_t vectorSize(5);
    constexpr size_t storageSize = size_t(2) * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (size_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      x(k) = vals.first;
      y(k) = vals.second;
    }

    linalg_copy(x, y);
    for (size_t k = 0; k < vectorSize; ++k) {
      const auto vals = makeVectorValues<scalar_t>(k);
      // Make sure the function didn't modify the input.
      EXPECT_EQ( x(k), vals.first );
      EXPECT_EQ( y(k), vals.first ); // check the output
    }
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
