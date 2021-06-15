#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

//#include <execution>
#include <vector>
#include "gtest/gtest.h"
#include "gtest_fixtures.hpp"

namespace {

  using std::experimental::linalg::vector_abs_sum;

  TEST_F(unsigned_double_vector, abs_sum)
  {
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR( 4.6, vector_abs_sum(v,  0.0), 1e-15);
    EXPECT_NEAR( 5.8, vector_abs_sum(v,  1.2), 1e-15);
    EXPECT_NEAR(-0.4, vector_abs_sum(v, -5.0), 1e-15);
    EXPECT_NEAR( 0.0, vector_abs_sum(v, -4.6), 1e-15);
  }

  TEST_F(signed_double_vector, abs_sum)
  {
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR( 4.6, vector_abs_sum(v,  0.0), 1e-15);
    EXPECT_NEAR( 5.8, vector_abs_sum(v,  1.2), 1e-15);
    EXPECT_NEAR(-0.4, vector_abs_sum(v, -5.0), 1e-15);
    EXPECT_NEAR( 0.0, vector_abs_sum(v, -4.6), 1e-15);
  }

  TEST_F(signed_complex_vector, abs_sum)
  {
    // EXPECT_DOUBLE_EQ expects values within 4 ULPs.
    // We don't get that accurate of an answer, so we use EXPECT_NEAR instead.
    EXPECT_NEAR(3.5188912597625004, vector_abs_sum(v, 0.0), 1e-15);
    EXPECT_NEAR(4.7188912597625004, vector_abs_sum(v, 1.2), 1e-15);
    EXPECT_NEAR(-0.4811087402374996, vector_abs_sum(v, -4.0), 1e-15);
    EXPECT_NEAR(0.0, vector_abs_sum(v, -3.5188912597625004), 1e-15);
  }

} // end anonymous namespace
