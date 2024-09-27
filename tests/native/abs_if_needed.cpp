#include "./my_numbers.hpp"

namespace {
  TEST(impl_abs_if_needed, arithmetic_types) {
    {
      auto input = 4u;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
    {
      auto input = -4;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, -input);
    }
    {
      auto input = 4;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
    {
      auto input = static_cast<unsigned long long>(4u);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
    {
      auto input = static_cast<long long>(-4);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, -input);
    }
    {
      auto input = static_cast<long long>(4);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
    {
      auto input = -5.0f;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, -input);
    }
    {
      auto input = 5.0f;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
    {
      auto input = -5.0;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, -input);
    }
    {
      auto input = 5.0;
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result, input);
    }
  }

  TEST(impl_abs_if_needed, custom_real) {
    {
      auto input = TestLinearAlgebra::MyReal(-4.0);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result.value(), -input.value());
    }
    {
      auto input = TestLinearAlgebra::MyReal(4.0);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), decltype(input)>);
      EXPECT_EQ(result.value(), input.value());
    }
  }

  TEST(impl_abs_if_needed, std_complex) {
    {
      auto input = std::complex<double>(-3.0, 4.0);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), double>);
      EXPECT_EQ(result, 5.0);
    }
    {
      auto input = std::complex<float>(0.0f, 2.0f);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), float>);
      EXPECT_EQ(result, 2.0f);
    }
  }

  TEST(impl_abs_if_needed, custom_complex) {
    {
      auto input = TestLinearAlgebra::MyComplex(-3.0, 4.0);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), double>);
      EXPECT_EQ(result, 5.0);
    }
    {
      auto input = TestLinearAlgebra::MyComplex(0.0, 2.0);
      auto result = LinearAlgebra::impl::abs_if_needed(input);
      static_assert(std::is_same_v<decltype(result), double>);
      EXPECT_EQ(result, 2.0);
    }
  }
} // end anonymous namespace
