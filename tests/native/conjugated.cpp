#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::conjugated;
  using LinearAlgebra::conjugated_accessor;

  // A clone of default_accessor, which we use to test the return type of conjugated.
  template<class ElementType>
  class nondefault_accessor {
  public:
    using reference        = ElementType&;
    using element_type     = ElementType;
    using data_handle_type = ElementType*;
    using offset_policy    = nondefault_accessor<ElementType>;

    constexpr nondefault_accessor() noexcept = default;
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherElementType,
      /* requires */ (std::is_convertible_v<OtherElementType(*)[], element_type(*)[]>)
    )
    nondefault_accessor(nondefault_accessor<OtherElementType> /* accessor */) noexcept
    {}

    reference access(data_handle_type p, ::std::size_t i) const noexcept
    {
      return p[i];
    }

    typename offset_policy::data_handle_type
      offset(data_handle_type p, ::std::size_t i) const noexcept
    {
      return p + i;
    }
  };

  TEST(conjugated, real_default_accessor)
  {
    std::array<float, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = default_accessor<float>;
      using expected_accessor_type = default_accessor<float>;
      mdspan<float, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);
      static_assert(std::is_same_v<decltype(x_nc_conj),
                    mdspan<float, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
    }
    {
      using input_accessor_type = default_accessor<const float>;
      using expected_accessor_type = default_accessor<const float>;
      mdspan<const float, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);
      static_assert(std::is_same_v<decltype(x_c_conj),
                    mdspan<const float, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
    }
  }

  TEST(conjugated, real_nondefault_accessor)
  {
    std::array<float, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = nondefault_accessor<float>;
      // Implementation currently is more like P3050R0 than P1673R13.
      using expected_accessor_type = nondefault_accessor<float>;
      mdspan<float, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);
      static_assert(std::is_same_v<decltype(x_nc_conj),
                    mdspan<float, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
    }
    {
      using input_accessor_type = nondefault_accessor<const float>;
      // Implementation currently is more like P3050R0 than P1673R13.
      using expected_accessor_type = nondefault_accessor<const float>;
      mdspan<const float, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);
      static_assert(std::is_same_v<decltype(x_c_conj),
                    mdspan<const float, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
    }
  }

  struct nonarithmetic_real {};

  // P3050 changes the behavior of conjugated for nonarithmetic "real" types.
  // "Real" means "types T for which conj<std::declval<T>()) is not ADL-findable."
  TEST(conjugated, nonarithmetic_real_default_accessor)
  {
    using value_type = nonarithmetic_real;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = default_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<default_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
                    mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
    }
    {
      using input_accessor_type = default_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<default_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
                    mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
    }
  }

  // P3050 changes the behavior of conjugated for nonarithmetic "real" types.
  // "Real" means "types T for which conj<std::declval<T>()) is not ADL-findable."
  TEST(conjugated, nonarithmetic_real_nondefault_accessor)
  {
    using value_type = nonarithmetic_real;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = nondefault_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
                    mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
    }
    {
      using input_accessor_type = nondefault_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
                    mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
    }
  }

  TEST(conjugated, complex_default_accessor)
  {
    using value_type = std::complex<double>;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = default_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<default_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
      static_assert(std::is_same_v<decltype(x_nc_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
    {
      using input_accessor_type = default_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<default_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
      static_assert(std::is_same_v<decltype(x_c_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
  }

  TEST(conjugated, complex_nondefault_accessor)
  {
    using value_type = std::complex<double>;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = nondefault_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
      static_assert(std::is_same_v<decltype(x_nc_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
    {
      using input_accessor_type = nondefault_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
      static_assert(std::is_same_v<decltype(x_c_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
  }

  struct custom_complex {
    friend custom_complex conj(const custom_complex& z) {
      return z;
    }
  };

  TEST(conjugated, impl_has_conj)
  {
    using MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: linalg::impl::has_conj;

    static_assert(! has_conj<int>::value);
    static_assert(! has_conj< ::std::size_t>::value);
    static_assert(! has_conj<float>::value);
    static_assert(! has_conj<double>::value);
    static_assert(! has_conj<nonarithmetic_real>::value);

    static_assert(has_conj<std::complex<float>>::value);
    static_assert(has_conj<std::complex<double>>::value);
    static_assert(has_conj<custom_complex>::value);
  }

  TEST(conjugated, custom_complex_default_accessor)
  {
    using value_type = custom_complex;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = default_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<default_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
      static_assert(std::is_same_v<decltype(x_nc_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
    {
      using input_accessor_type = default_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<default_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
      static_assert(std::is_same_v<decltype(x_c_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
  }

  TEST(conjugated, custom_complex_nondefault_accessor)
  {
    using value_type = custom_complex;
    std::array<value_type, 3> x_storage{};
    using extents_type = extents<int, 3>;
    using layout_type = layout_right;

    {
      using input_accessor_type = nondefault_accessor<value_type>;
      mdspan<value_type, extents_type, layout_type, input_accessor_type> x_nc{x_storage.data()};
      auto x_nc_conj = conjugated(x_nc);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_nc_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_nc_conj.mapping(), x_nc.mapping());
      static_assert(std::is_same_v<decltype(x_nc_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
    {
      using input_accessor_type = nondefault_accessor<const value_type>;
      mdspan<const value_type, extents_type, layout_type, input_accessor_type> x_c{x_storage.data()};
      auto x_c_conj = conjugated(x_c);

      using expected_accessor_type = conjugated_accessor<nondefault_accessor<const value_type>>;
      using expected_element_type = std::add_const_t<value_type>;
      static_assert(std::is_same_v<decltype(x_c_conj),
        mdspan<expected_element_type, extents_type, layout_type, expected_accessor_type>>);
      EXPECT_EQ(x_c_conj.mapping(), x_c.mapping());
      static_assert(std::is_same_v<decltype(x_c_conj.accessor().nested_accessor()), const input_accessor_type&>);
    }
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

    // Make sure that conjugated_accessor compiles
    {
      using accessor_t = vector_t::accessor_type;
      using accessor_conj_t = conjugated_accessor<accessor_t>;
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
