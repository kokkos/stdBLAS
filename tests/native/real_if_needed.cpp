#include "./my_numbers.hpp"

namespace {
  template<class R>
  void test_real_if_needed_complex()
  {
    using LinearAlgebra::impl::real_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = real_if_needed(z);
    EXPECT_EQ(z_imag, R(3.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_real_if_needed_floating_point()
  {
    using LinearAlgebra::impl::real_if_needed;
    T x = 9.0;
    auto x_imag = real_if_needed(x);
    EXPECT_EQ(x_imag, T(9.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_real_if_needed_integral()
  {
    using LinearAlgebra::impl::real_if_needed;
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
      using LinearAlgebra::impl::real_if_needed;
      TestLinearAlgebra::MyComplex z{ 3.0, 4.0 };
      auto z_imag = real_if_needed(z);
      EXPECT_EQ(z_imag, 3.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(imag(z))>);
    }
    {
      using LinearAlgebra::impl::real_if_needed;
      TestLinearAlgebra::MyReal x{ 3.0 };
      auto x_real = real_if_needed(x);
      EXPECT_EQ(x_real, TestLinearAlgebra::MyReal{ 3.0 });
      static_assert(std::is_same_v<decltype(x_real), TestLinearAlgebra::MyReal>);
    }
  }
} // end anonymous namespace
