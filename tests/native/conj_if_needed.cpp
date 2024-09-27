#include "./my_numbers.hpp"

namespace {
  template<class Real>
  void test_real_conj_if_needed()
  {
    using LinearAlgebra::impl::conj_if_needed;

    Real z(2.0);
    const Real z_conj_expected(2.0);

    auto z_conj = conj_if_needed(z);
    static_assert(std::is_same_v<decltype(z_conj), Real>);
    EXPECT_EQ(z_conj, z_conj_expected);
  }

  template<class Complex>
  void test_any_complex_conj_if_needed()
  {
    using LinearAlgebra::impl::conj_if_needed;

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

  void test_MyComplex_conj_if_needed()
  {
    test_any_complex_conj_if_needed<TestLinearAlgebra::MyComplex>();
  }

  TEST(test_numbers, conj_if_needed)
  {
    test_std_complex_conj_if_needed<float>();
    test_std_complex_conj_if_needed<double>();
    test_std_complex_conj_if_needed<long double>();

    test_MyComplex_conj_if_needed();

    test_real_conj_if_needed<float>();
    test_real_conj_if_needed<double>();
    test_real_conj_if_needed<long double>();

    test_real_conj_if_needed<int32_t>();
    test_real_conj_if_needed<uint32_t>();
    test_real_conj_if_needed<int64_t>();
    test_real_conj_if_needed<uint64_t>();
  }
} // end anonymous namespace
