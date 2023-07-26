#define MDSPAN_USE_PAREN_OPERATOR 1
#define P1673_CONJUGATED_SCALAR_ARITHMETIC_OPERATORS_REFERENCE_OVERLOADS 1

#include "gtest/gtest.h"
#include "./test_numbers.hpp"
#include <experimental/linalg>

// Specialize test helper traits (P1673 does NOT need these)
namespace test_helpers {

template<class T>
constexpr bool is_complex_v = false;

template<>
constexpr bool is_complex_v<std::complex<float>> = true;

template<>
constexpr bool is_complex_v<std::complex<double>> = true;

template<>
constexpr bool is_complex_v<std::complex<long double>> = true;

template<>
constexpr bool is_complex_v<FakeComplex> = true;

template<class T>
static constexpr bool is_atomic_ref_not_arithmetic_v = false;

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
template<class U>
static constexpr bool is_atomic_ref_not_arithmetic_v<std::atomic_ref<U>> = ! std::is_arithmetic_v<U>;
#endif

} // namespace test_helpers

///////////////////////////////////////////////////////////
// conj_if_needed tests
///////////////////////////////////////////////////////////

namespace {
  template<class Real>
  void test_real_conj_if_needed()
  {
    using std::experimental::linalg::impl::conj_if_needed;

    Real z(2.0);
    const Real z_conj_expected(2.0);

    auto z_conj = conj_if_needed(z);
    static_assert(std::is_same_v<decltype(z_conj), Real>);
    EXPECT_EQ(z_conj, z_conj_expected);
  }

  template<class Complex>
  void test_any_complex_conj_if_needed()
  {
    using std::experimental::linalg::impl::conj_if_needed;

    Complex z(2.0, -3.0);
    Complex z_orig(2.0, -3.0);
    const Complex z_conj_expected(2.0, 3.0);

    auto z_conj = conj_if_needed(z);
    static_assert(std::is_same_v<decltype(z_conj), Complex>);
    EXPECT_EQ(z_conj, z_conj_expected);
    EXPECT_EQ(z, z_orig); // conj didn't change its input
  }

  template<class Real>
  void test_std_complex_conj_if_needed()
  {
    test_any_complex_conj_if_needed<std::complex<Real>>();
  }

  void test_FakeComplex_conj_if_needed()
  {
    test_any_complex_conj_if_needed<FakeComplex>();
  }

  TEST(test_numbers, conj_if_needed)
  {
    test_std_complex_conj_if_needed<float>();
    test_std_complex_conj_if_needed<double>();
    test_std_complex_conj_if_needed<long double>();

    test_FakeComplex_conj_if_needed();

    test_real_conj_if_needed<float>();
    test_real_conj_if_needed<double>();
    test_real_conj_if_needed<long double>();

    test_real_conj_if_needed<int32_t>();
    test_real_conj_if_needed<uint32_t>();
    test_real_conj_if_needed<int64_t>();
    test_real_conj_if_needed<uint64_t>();
  }

  template<class R>
  void test_imag_if_needed_complex()
  {
    using std::experimental::linalg::impl::imag_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = imag_if_needed(z);
    EXPECT_EQ(z_imag, R(4.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_imag_if_needed_floating_point()
  {
    using std::experimental::linalg::impl::imag_if_needed;
    T x = 9.0;
    auto x_imag = imag_if_needed(x);
    EXPECT_EQ(x_imag, T(0.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_imag_if_needed_integral()
  {
    using std::experimental::linalg::impl::imag_if_needed;
    T x = 3;
    auto x_imag = imag_if_needed(x);
    EXPECT_EQ(x_imag, T(0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }

  TEST(test_numbers, imag_if_needed)
  {
    test_imag_if_needed_complex<float>();
    test_imag_if_needed_complex<double>();
    test_imag_if_needed_complex<long double>();
    
    test_imag_if_needed_floating_point<float>();
    test_imag_if_needed_floating_point<double>();
    test_imag_if_needed_floating_point<long double>();

    test_imag_if_needed_integral<int8_t>();
    test_imag_if_needed_integral<uint8_t>();
    test_imag_if_needed_integral<int16_t>();
    test_imag_if_needed_integral<uint16_t>();
    test_imag_if_needed_integral<int32_t>();
    test_imag_if_needed_integral<uint32_t>();
    test_imag_if_needed_integral<int64_t>();
    test_imag_if_needed_integral<uint64_t>();

    {
      using std::experimental::linalg::impl::imag_if_needed;
      FakeComplex z{3.0, 4.0};
      auto z_imag = imag_if_needed(z);
      EXPECT_EQ(z_imag, 4.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(z.imag)>);
    }
    {
      using std::experimental::linalg::impl::imag_if_needed;
      FakeRealNumber x{3.0};
      auto x_imag = imag_if_needed(x);
      EXPECT_EQ(x_imag, FakeRealNumber{});
      static_assert(std::is_same_v<decltype(x_imag), FakeRealNumber>);
    }
  }

  template<class R>
  void test_real_if_needed_complex()
  {
    using std::experimental::linalg::impl::real_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = real_if_needed(z);
    EXPECT_EQ(z_imag, R(3.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_real_if_needed_floating_point()
  {
    using std::experimental::linalg::impl::real_if_needed;
    T x = 9.0;
    auto x_imag = real_if_needed(x);
    EXPECT_EQ(x_imag, T(9.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_real_if_needed_integral()
  {
    using std::experimental::linalg::impl::real_if_needed;
    T x = 3;
    auto x_imag = real_if_needed(x);
    EXPECT_EQ(x_imag, T(3));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }

  TEST(test_numbers, real_if_needed)
  {
    test_real_if_needed_complex<float>();
    test_real_if_needed_complex<double>();
    test_real_if_needed_complex<long double>();

    test_real_if_needed_floating_point<float>();
    test_real_if_needed_floating_point<double>();
    test_real_if_needed_floating_point<long double>();

    test_real_if_needed_integral<int8_t>();
    test_real_if_needed_integral<uint8_t>();
    test_real_if_needed_integral<int16_t>();
    test_real_if_needed_integral<uint16_t>();
    test_real_if_needed_integral<int32_t>();
    test_real_if_needed_integral<uint32_t>();
    test_real_if_needed_integral<int64_t>();
    test_real_if_needed_integral<uint64_t>();

    {
      using std::experimental::linalg::impl::real_if_needed;
      FakeComplex z{ 3.0, 4.0 };
      auto z_imag = real_if_needed(z);
      EXPECT_EQ(z_imag, 3.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(z.imag)>);
    }
    {
      using std::experimental::linalg::impl::real_if_needed;
      FakeRealNumber x{ 3.0 };
      auto x_real = real_if_needed(x);
      EXPECT_EQ(x_real, FakeRealNumber{ 3.0 });
      static_assert(std::is_same_v<decltype(x_real), FakeRealNumber>);
    }
  }
} // namespace (anonymous)
