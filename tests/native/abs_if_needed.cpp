#include "./gtest_fixtures.hpp"

namespace TestLinearAlgebra {

class MyReal {
public:
  MyReal() = default;
  explicit MyReal(double value) : value_(value) {}
  double value() const { return value_; }

  friend MyReal abs(MyReal x) { return MyReal{std::abs(x.value())}; }

private:
  double value_ = 0.0;
};

} // namespace TestLinearAlgebra

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

} // end anonymous namespace
