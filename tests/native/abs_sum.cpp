#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::vector_abs_sum;

  class unsigned_double_vector : public ::testing::Test {
  private:
    std::vector<double> storage{
      0.03125,
      0.0625,
      0.125,
      0.25,
      0.5,
      1.0,
      2.0,
      4.0,
      8.0,
      16.0,
      32.0
    };
  protected:
    using mdspan_type =
      mdspan<const double, extents<std::size_t, dynamic_extent>>;

    mdspan_type get_test_mdspan() const {
      return mdspan_type{storage.data(), 11};
    }

    double expected_abs_sum() const {
      return 63.96875; // 63 and 31/32
    }
  };

  class signed_double_vector : public ::testing::Test {
  private:
    std::vector<double> storage{
      -0.03125,
      0.0625,
      -0.125,
      0.25,
      -0.5,
      1.0,
      -2.0,
      4.0,
      -8.0,
      16.0,
      -32.0
    };
  protected:
    using mdspan_type =
      mdspan<const double, extents<std::size_t, dynamic_extent>>;

    mdspan_type get_test_mdspan() const {
      return mdspan_type{storage.data(), 11};
    }

    double expected_abs_sum() const {
      return 63.96875; // 63 and 31/32
    }
  };

  class signed_complex_vector : public ::testing::Test {
  private:
    std::vector<std::complex<double>> storage{
      { -0.03125,  0.0625},
      {  0.0625,  -0.125},
      { -0.125,    0.25},
      {  0.25,    -0.5},
      { -0.5,      1.0},
      {  1.0,     -2.0},
      { -2.0,      4.0},
      {  4.0,     -8.0},
      { -8.0,     16.0},
      { 16.0,    -32.0},
      {-32.0,     64.0}
    };
  protected:
    using mdspan_type =
      mdspan<const std::complex<double>,
        extents<std::size_t, dynamic_extent>>;

    mdspan_type get_test_mdspan() const {
      return mdspan_type{storage.data(), 11};
    }

    double expected_abs_sum() const {
      return 191.90625; // 63 and 31/32, plus 127 and 15/16
    }
  };

  TEST_F(unsigned_double_vector, abs_sum)
  {
    auto v = this->get_test_mdspan();
    auto expected = this->expected_abs_sum();
    const auto tol =
      std::sqrt(expected) * std::numeric_limits<decltype(expected)>::epsilon();

    EXPECT_NEAR( expected,          vector_abs_sum(v,  0.0),      tol);
    EXPECT_NEAR( (expected + 1.25), vector_abs_sum(v,  1.25),     tol);
    EXPECT_NEAR( (expected - 5.0),  vector_abs_sum(v, -5.0),      tol);
    EXPECT_NEAR( 0.0,               vector_abs_sum(v, -expected), tol);

    // Test 'auto' overload.
    auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<decltype(sumResultAuto), decltype(expected)> );
    EXPECT_NEAR( expected, sumResultAuto, tol );
  }

  TEST_F(signed_double_vector, abs_sum)
  {
    auto v = this->get_test_mdspan();
    auto expected = this->expected_abs_sum();
    const auto tol =
      std::sqrt(expected) * std::numeric_limits<decltype(expected)>::epsilon();

    EXPECT_NEAR( expected,          vector_abs_sum(v,  0.0),      tol);
    EXPECT_NEAR( (expected + 1.25), vector_abs_sum(v,  1.25),     tol);
    EXPECT_NEAR( (expected - 5.0),  vector_abs_sum(v, -5.0),      tol);
    EXPECT_NEAR( 0.0,               vector_abs_sum(v, -expected), tol);

    // Test 'auto' overload.
    auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<decltype(sumResultAuto), decltype(expected)> );
    EXPECT_NEAR( expected, sumResultAuto, tol );
  }

  TEST_F(signed_complex_vector, abs_sum)
  {
    auto v = this->get_test_mdspan();
    auto expected = this->expected_abs_sum();
    // Complex numbers have two parts (real and imaginary).
    // This means there are twice as many terms in the sum.
    const auto tol =
      std::sqrt(decltype(expected)(2.0) * expected) *
      std::numeric_limits<decltype(expected)>::epsilon();

    EXPECT_NEAR( expected,          vector_abs_sum(v,  0.0),      tol);
    EXPECT_NEAR( (expected + 1.25), vector_abs_sum(v,  1.25),     tol);
    EXPECT_NEAR( (expected - 5.0),  vector_abs_sum(v, -5.0),      tol);
    EXPECT_NEAR( 0.0,               vector_abs_sum(v, -expected), tol);

    // Test 'auto' overload.
    auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<decltype(sumResultAuto), decltype(expected)> );
    EXPECT_NEAR( expected, sumResultAuto, tol );
  }

} // end anonymous namespace
