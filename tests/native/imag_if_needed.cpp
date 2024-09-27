#include "./my_numbers.hpp"

namespace {
  template<class R>
  void test_imag_if_needed_complex()
  {
    using LinearAlgebra::impl::imag_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = imag_if_needed(z);
    EXPECT_EQ(z_imag, R(4.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_imag_if_needed_floating_point()
  {
    using LinearAlgebra::impl::imag_if_needed;
    T x = 9.0;
    auto x_imag = imag_if_needed(x);
    EXPECT_EQ(x_imag, T(0.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_imag_if_needed_integral()
  {
    using LinearAlgebra::impl::imag_if_needed;
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
      using LinearAlgebra::impl::imag_if_needed;
      TestLinearAlgebra::MyComplex z{3.0, 4.0};
      auto z_imag = imag_if_needed(z);
      EXPECT_EQ(z_imag, 4.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(imag(z))>);
    }
    {
      using LinearAlgebra::impl::imag_if_needed;
      TestLinearAlgebra::MyReal x{3.0};
      auto x_imag = imag_if_needed(x);
      EXPECT_EQ(x_imag, TestLinearAlgebra::MyReal{});
      static_assert(std::is_same_v<decltype(x_imag), TestLinearAlgebra::MyReal>);
    }
  }
} // end anonymous namespace
