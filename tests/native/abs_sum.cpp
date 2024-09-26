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
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR(3.5188912597625004, vector_abs_sum(v, 0.0), 1e-15);
    EXPECT_NEAR(4.7188912597625004, vector_abs_sum(v, 1.2), 1e-15);
    EXPECT_NEAR(-0.4811087402374996, vector_abs_sum(v, -4.0), 1e-15);
    EXPECT_NEAR(0.0, vector_abs_sum(v, -3.5188912597625004), 1e-15);

    // Test 'auto' overload.
    const auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<std::remove_const_t<decltype(sumResultAuto)>, double> );
    EXPECT_NEAR( 3.5188912597625004, sumResultAuto, 1e-15 );
  }

} // end anonymous namespace
