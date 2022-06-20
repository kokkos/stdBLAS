#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <complex>
#include <vector>

namespace {
  using std::experimental::dextents;
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::conjugated;

  template<class ValueType>
  void test_accessor_conjugate_element_constification()
  {
    using std::experimental::linalg::accessor_conjugate;
    using std::experimental::default_accessor;
    using std::experimental::linalg::conjugated_scalar;
    using value_type = std::remove_cv_t<ValueType>;
    constexpr bool is_arith = std::is_arithmetic_v<value_type>;

    using nc_def_acc_type = default_accessor<value_type>;
    using c_def_acc_type = default_accessor<const value_type>;
    nc_def_acc_type nc_acc;
    c_def_acc_type c_acc;

    using aj_nc_type = accessor_conjugate<nc_def_acc_type>;
    using expected_nc_ref =
      std::conditional_t<is_arith, value_type&,
			 conjugated_scalar<value_type&, value_type>>;
    static_assert(std::is_same_v<expected_nc_ref, typename aj_nc_type::reference>);
    using expected_nc_elt = std::add_const_t<std::conditional_t<is_arith, value_type, typename conjugated_scalar<value_type&, value_type>::value_type>>;
    static_assert(std::is_same_v<expected_nc_elt, typename aj_nc_type::element_type>);
    static_assert(std::is_same_v<typename aj_nc_type::pointer, value_type*>);

    using aj_c_type = accessor_conjugate<c_def_acc_type>;
    using expected_c_ref = std::conditional_t<is_arith, const value_type&, conjugated_scalar<const value_type&, value_type>>;
    static_assert(std::is_same_v<expected_c_ref, typename aj_c_type::reference>);
    using expected_c_elt = std::add_const_t<std::conditional_t<is_arith, const value_type, typename conjugated_scalar<const value_type&, value_type>::value_type>>;
    static_assert(std::is_same_v<expected_c_elt, typename aj_c_type::element_type>);
    static_assert(std::is_same_v<typename aj_c_type::pointer, const value_type*>);

    aj_nc_type acc_conj_nc(nc_acc);
    aj_c_type acc_conj_c0(c_acc);

    // Test element_type constification (converting) constructor
    aj_c_type acc_conj_c1(nc_acc);
  }

  TEST(accessor_conjugate, element_constification)
  {
    test_accessor_conjugate_element_constification<double>();
    test_accessor_conjugate_element_constification<int>();
    test_accessor_conjugate_element_constification<std::complex<double>>();
    test_accessor_conjugate_element_constification<std::complex<float>>();
  }

  TEST(conjugated, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, dextents<std::size_t, 1>>;

    constexpr std::size_t vectorSize (5);
    constexpr std::size_t storageSize = std::size_t (2) * vectorSize;
    std::vector<scalar_t> storage (storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      const scalar_t y_k(real_t(k) + 2.0, real_t(k) + 2.0);
      x(k) = x_k;
      y(k) = y_k;
    }

    // Make sure that accessor_conjugate compiles
    {
      using accessor_t = vector_t::accessor_type;
      using std::experimental::linalg::accessor_conjugate;
      using accessor_conj_t = accessor_conjugate<accessor_t>;
      accessor_conj_t acc{y.accessor()};
    }

    auto y_conj = conjugated(y);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 1.0, real_t(k) + 1.0);
      EXPECT_EQ( x(k), x_k );

      // Make sure that conjugated doesn't modify the entries of
      // the original thing.
      const scalar_t y_k (real_t(k) + 2.0, real_t(k) + 2.0);
      EXPECT_EQ( y(k), y_k );

      const scalar_t y_k_conj (real_t(k) + 2.0, -real_t(k) - 2.0);
      EXPECT_EQ( scalar_t(y_conj(k)), y_k_conj );
    }
  }
}
