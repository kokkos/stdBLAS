#include <experimental/linalg>
#include <experimental/mdspan>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::basic_mdspan;
  using std::experimental::linalg::scaled;

  TEST(scaled, mdspan_double_scalar_float)
  {
    using vector_element_type = double;
    using scaling_factor_type = float;

    using vector_t =
      basic_mdspan<vector_element_type, extents<dynamic_extent>>;

    constexpr ptrdiff_t vectorSize (5);
    constexpr ptrdiff_t storageSize = ptrdiff_t (2) * vectorSize;
    std::vector<vector_element_type> storage (storageSize);

    vector_t x (storage.data (), vectorSize);
    vector_t y (storage.data () + vectorSize, vectorSize);

    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const vector_element_type x_k = vector_element_type(k) + 1.0;
      const vector_element_type y_k = vector_element_type(k) + 2.0;
      x(k) = x_k;
      y(k) = y_k;
    }

    const scaling_factor_type scalingFactor (-3.0);

    // Make sure that accessor_scaled compiles
    {
      using accessor_t = vector_t::accessor_type;
      using std::experimental::linalg::accessor_scaled;
      using scaled_accessor_t =
        accessor_scaled<accessor_t, scaling_factor_type>;
      scaled_accessor_t accessor0;
      scaled_accessor_t accessor1 (y.accessor (), scalingFactor);
    }

    auto y_scaled = scaled (scalingFactor, y);
    for (ptrdiff_t k = 0; k < vectorSize; ++k) {
      const vector_element_type x_k = vector_element_type(k) + 1.0;
      EXPECT_EQ( x(k), x_k );

      // Make sure that scaled doesn't modify the entries of the
      // original thing.
      const vector_element_type y_k = vector_element_type(k) + 2.0;
      EXPECT_EQ( y(k), y_k );

      const vector_element_type y_k_scaled = scalingFactor * y_k;
      EXPECT_EQ( y_scaled(k), y_k_scaled );

      // Don't ever capture an expression template type by auto in
      // real code.  I'm just testing whether some operators work.
      auto y_scaled_ref = y_scaled(k);
      using ref_t = decltype(y_scaled_ref);
      static_assert(! std::is_same_v<ref_t, vector_element_type>);

      using type1 = decltype(y_scaled_ref + float(1.0));
      static_assert(std::is_same_v<type1, double>);

      long double one_ld (1.0);
      static_assert(std::is_same_v<decltype(one_ld), long double>);
      using type2 = decltype(y_scaled_ref + one_ld);
      static_assert(std::is_same_v<type2, long double>);

      EXPECT_EQ( -y_scaled_ref, -y_k_scaled );
      EXPECT_EQ( y_scaled_ref + 418.0, y_k_scaled + 418.0 );
      EXPECT_EQ( 666.0 + y_scaled_ref, 666.0 + y_k_scaled );
      EXPECT_EQ( -2.0 * y_scaled_ref, -2.0 * y_k_scaled );
      EXPECT_EQ( y_scaled_ref * -2.0, y_k_scaled * -2.0 );
      EXPECT_EQ( 1.0 / y_scaled_ref, 1.0 / y_k_scaled );
    }
  }
}
