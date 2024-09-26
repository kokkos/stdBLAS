#include "./gtest_fixtures.hpp"
#include <experimental/linalg>
#include <atomic>

// Proxy reference type for tests.
template<class ElementType>
class my_proxy_reference {
private:
  using this_type = my_proxy_reference<ElementType>;
  using element_type = ElementType;
  using value_type = std::remove_cv_t<element_type>;
  using reference = element_type&;

  element_type* ptr_;

public:
  explicit my_proxy_reference(reference ref) : ptr_(&ref) {}
  my_proxy_reference(const my_proxy_reference& shallow_copy) : ptr_(shallow_copy.ptr_) {}
  // my_proxy_reference is not copy assignable.
  my_proxy_reference& operator=(const my_proxy_reference&) = delete;

  value_type operator=(value_type desired) const noexcept {
    *ptr_ = desired;
    return desired;
  }

  operator value_type() const noexcept {
    return value_type(*ptr_);
  }
};

template <class ElementType>
struct my_proxy_reference_accessor {
  using offset_policy = my_proxy_reference_accessor;
  using element_type = ElementType;
  using reference = my_proxy_reference<element_type>;
  using data_handle_type = ElementType*;

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr my_proxy_reference_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, OtherElementType(*)[], element_type(*)[])
      )
  )
    MDSPAN_INLINE_FUNCTION
    constexpr my_proxy_reference_accessor(my_proxy_reference_accessor<OtherElementType>) noexcept {}

  MDSPAN_INLINE_FUNCTION
    constexpr data_handle_type
    offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
    constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return reference{ p[i] };
  }
};

#if defined(__cpp_lib_atomic_ref)
// P2689 proposes atomic_accessor.  This is a simpler version.
// The point of this is to have an accessor whose "reference"
// is a proxy reference type, instead of element_type&.
// In this case, the proxy reference type is atomic_ref,
// a C++20 feature.
//
// We want an accessor with a proxy reference
// because we want to test arithmetic expressions
// involving mdspan accesses that mix element_type
// and a proxy reference type.
//
// atomic_ref<T> has overloaded arithmetic operators
// when T is an arithmetic type.  It turns out we need
// those overloaded arithmetic operators.  We can't ask mdspan
// to work miracles; the user's proxy reference type
// needs to have all the arithmetic operators needed
// for the arithmetic expressions that occur in the program.
template <class ElementType>
struct atomic_accessor {
  using offset_policy = atomic_accessor;
  using element_type = ElementType;
  using reference = std::atomic_ref<ElementType>;
  using data_handle_type = ElementType*;

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr atomic_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, OtherElementType(*)[], element_type(*)[])
      )
  )
    MDSPAN_INLINE_FUNCTION
    constexpr atomic_accessor(atomic_accessor<OtherElementType>) noexcept {}

  MDSPAN_INLINE_FUNCTION
    constexpr data_handle_type
    offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
    constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return reference{ p[i] };
  }
};
#endif // __cpp_lib_atomic_ref

namespace {
  using LinearAlgebra::conjugated;
  using LinearAlgebra::scaled;

  TEST(mixed_accessors, mdspan_scaled_and_my_proxy)
  {
    using real_t = int;
    using scalar_t = int;
    using extents_t = dextents<std::size_t, 1>;
    using vector_t = mdspan<scalar_t, extents_t>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = 2 * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 11);
      const scalar_t y_k(real_t(k) + 5);
      x(k) = x_k;
      y(k) = y_k;
    }

    mdspan<scalar_t, extents_t, layout_right, my_proxy_reference_accessor<scalar_t>> x_proxy(x.data_handle(), x.mapping());
    const int alpha = 2;
    auto y_scaled = scaled(alpha, y);
    for (std::size_t k = 0; k < vectorSize; ++k) {
      ASSERT_EQ(x_proxy(k), real_t(k) + 11);
      ASSERT_EQ(y_scaled(k), 2 * real_t(k) + 10);

      auto x_minus_y = x_proxy(k) - y_scaled(k);
      static_assert(std::is_same_v<decltype(x_minus_y), scalar_t>);
      EXPECT_EQ(x_minus_y, -real_t(k) + 1);

      auto y_minus_x = y_scaled(k) - x_proxy(k);
      static_assert(std::is_same_v<decltype(y_minus_x), scalar_t>);
      EXPECT_EQ(y_minus_x, real_t(k) - 1);

      auto minus_y = -y_scaled(k);
      static_assert(std::is_same_v<decltype(minus_y), scalar_t>);
      EXPECT_EQ(minus_y, -2 * real_t(k) - 10);

      auto minus_x = -x_proxy(k);
      static_assert(std::is_same_v<decltype(minus_x), scalar_t>);
      EXPECT_EQ(minus_x, -real_t(k) - 11);
    }
  }

#if defined(__cpp_lib_atomic_ref)
  TEST(mixed_accessors, mdspan_scaled_and_atomic)
  {
    using real_t = float;
    using scalar_t = real_t;
    using extents_t = dextents<std::size_t, 1>;
    using vector_t = mdspan<scalar_t, extents_t>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = 2 * vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);
    vector_t y(storage.data() + vectorSize, vectorSize);

    for (std::size_t k = 0; k < vectorSize; ++k) {
      const scalar_t x_k(real_t(k) + 5.0f);
      const scalar_t y_k(real_t(k) + 11.0f);
      x(k) = x_k;
      y(k) = y_k;
    }

    const scalar_t alpha = 2.0f;
    auto x_scaled = scaled(alpha, x);
    mdspan<scalar_t, extents_t, layout_right, atomic_accessor<scalar_t>> y_proxy(y.data_handle(), y.mapping());
    for (std::size_t k = 0; k < vectorSize; ++k) {
      ASSERT_EQ(x_scaled(k), (alpha * (real_t(k) + 5.0f)));
      ASSERT_EQ(y_proxy(k), (real_t(k) + 11.0f));

      auto y_minus_x = y_proxy(k) - x_scaled(k);
      static_assert(std::is_same_v<decltype(y_minus_x), scalar_t>);
      const scalar_t y_minus_x_expected = -real_t(k) + 1.0f;
      EXPECT_EQ(y_minus_x, y_minus_x_expected);

      auto x_minus_y = x_scaled(k) - y_proxy(k);
      static_assert(std::is_same_v<decltype(x_minus_y), scalar_t>);
      const scalar_t x_minus_y_expected = real_t(k) - 1.0f;
      EXPECT_EQ(x_minus_y, x_minus_y_expected);

      auto minus_y = -y_proxy(k);
      static_assert(std::is_same_v<decltype(minus_y), scalar_t>);
      const scalar_t minus_y_expected(-real_t(k) - 11.0f);
      EXPECT_EQ(minus_y, minus_y_expected);

      auto minus_x = -x_scaled(k);
      static_assert(std::is_same_v<decltype(minus_x), scalar_t>);
      const scalar_t minus_x_expected(-2.0f * real_t(k) - 10.0f);
      EXPECT_EQ(minus_x, minus_x_expected);
    }
  }
#endif // __cpp_lib_atomic_ref

  TEST(mixed_accessors, mdspan_scaled_and_conjugated)
  {
    using real_t = float;
    using complex_t = std::complex<real_t>;
    using extents_t = dextents<std::size_t, 1>;
    using real_vector_t = mdspan<real_t, extents_t>;
    using complex_vector_t = mdspan<complex_t, extents_t>;

    constexpr std::size_t vector_size(5);
    std::vector<real_t> real_storage(vector_size);
    std::vector<complex_t> complex_storage(vector_size);

    real_vector_t x(real_storage.data(), vector_size);
    complex_vector_t y(complex_storage.data(), vector_size);

    for (std::size_t k = 0; k < vector_size; ++k) {
      const real_t x_k(real_t(k) + 5.0f);
      const complex_t y_k(real_t(k) + 11.0f, real_t(k) + 13.0f);;
      x(k) = x_k;
      y(k) = y_k;
    }

    const real_t alpha = 2.0f;
    auto x_scaled = scaled(alpha, x);
    auto y_conj = conjugated(y);

    for (std::size_t k = 0; k < vector_size; ++k) {
      ASSERT_EQ(x_scaled(k), (2.0f * (real_t(k) + 5.0f)));
      ASSERT_EQ(y_conj(k), (complex_t(real_t(k) + 11.0f, -real_t(k) - 13.0f)));

      auto y_minus_x = y_conj(k) - x_scaled(k);
      static_assert(std::is_same_v<decltype(y_minus_x), complex_t>);
      const complex_t y_minus_x_expected(-real_t(k) + 1.0f, -real_t(k) - 13.0f);
      EXPECT_EQ(y_minus_x, y_minus_x_expected);

      auto x_minus_y = x_scaled(k) - y_conj(k);
      static_assert(std::is_same_v<decltype(x_minus_y), complex_t>);
      const complex_t x_minus_y_expected(real_t(k) - 1.0f, real_t(k) + 13.0f);
      EXPECT_EQ(x_minus_y, x_minus_y_expected);

      auto minus_y = -y_conj(k);
      static_assert(std::is_same_v<decltype(minus_y), complex_t>);
      const complex_t minus_y_expected(-real_t(k) - 11.0f, real_t(k) + 13.0f);
      EXPECT_EQ(minus_y, minus_y_expected);

      auto minus_x = -x_scaled(k);
      static_assert(std::is_same_v<decltype(minus_x), real_t>);
      const real_t minus_x_expected(-2.0f * real_t(k) - 10.0f);
      EXPECT_EQ(minus_x, minus_x_expected);
    }
  }
} // namespace (anonymous)
