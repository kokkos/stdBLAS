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

  using std::experimental::linalg::idx_abs_max;

  TEST_F(unsigned_double_vector, idx_abs_max)
  {
    EXPECT_EQ(9, idx_abs_max(v));
  }

  TEST_F(signed_double_vector, idx_abs_max)
  {
    EXPECT_EQ(9, idx_abs_max(v));
  }

  TEST_F(signed_complex_vector, idx_abs_max)
  {
    EXPECT_EQ(2, idx_abs_max(v));
  }

} // end anonymous namespace
