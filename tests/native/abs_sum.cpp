#include "./gtest_fixtures.hpp"

namespace {

  using LinearAlgebra::vector_abs_sum;

  TEST_F(unsigned_double_vector, abs_sum)
  {
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR( 4.6, vector_abs_sum(v,  0.0), 1e-15);
    EXPECT_NEAR( 5.8, vector_abs_sum(v,  1.2), 1e-15);
    EXPECT_NEAR(-0.4, vector_abs_sum(v, -5.0), 1e-15);
    EXPECT_NEAR( 0.0, vector_abs_sum(v, -4.6), 1e-15);

    // Test 'auto' overload.
    const auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<std::remove_const_t<decltype(sumResultAuto)>, double> );
    EXPECT_NEAR( 4.6, sumResultAuto, 1e-15 );
  }

  TEST_F(signed_double_vector, abs_sum)
  {
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR( 4.6, vector_abs_sum(v,  0.0), 1e-15);
    EXPECT_NEAR( 5.8, vector_abs_sum(v,  1.2), 1e-15);
    EXPECT_NEAR(-0.4, vector_abs_sum(v, -5.0), 1e-15);
    EXPECT_NEAR( 0.0, vector_abs_sum(v, -4.6), 1e-15);

    // Test 'auto' overload.
    const auto sumResultAuto = vector_abs_sum(v);
    static_assert( std::is_same_v<std::remove_const_t<decltype(sumResultAuto)>, double> );
    EXPECT_NEAR( 4.6, sumResultAuto, 1e-15 );
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
