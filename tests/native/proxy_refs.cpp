#define P1673_CONJUGATED_SCALAR_ARITHMETIC_OPERATORS_REFERENCE_OVERLOADS 1

#include "./gtest_fixtures.hpp"
#include "experimental/__p1673_bits/proxy_reference.hpp"

///////////////////////////////////////////////////////////
// Custom real number type for tests
///////////////////////////////////////////////////////////

struct FakeRealNumber {
    float value;

#ifdef __cpp_impl_three_way_comparison
    bool operator==(const FakeRealNumber&) const = default;
#else
  friend bool operator==(const FakeRealNumber& x, const FakeRealNumber& y)
  {
    return x.value == y.value;
  }
#endif
};

// Custom complex number type

class FakeComplex {
private:
  double real_ = 0.0;
  double imag_ = 0.0;

public:
  FakeComplex() = default;
  FakeComplex(double re, double im) : real_(re), imag_(im) {}

  friend double real(FakeComplex z) {
    return z.real_;
  }

  friend double imag(FakeComplex z) {
    return z.imag_;
  }

  friend double abs(FakeComplex z) {
    return std::sqrt(z.real_ * z.real_ + z.imag_ * z.imag_);
  }

  friend FakeComplex conj(FakeComplex z) {
    return {z.real_, -z.imag_};
  }

#ifdef __cpp_impl_three_way_comparison
  bool operator==(const FakeComplex&) const = default;
#else
  friend bool operator==(const FakeComplex& x, const FakeComplex& y)
  {
    return x.real_ == y.real_ && x.imag_ == y.imag_;
  }
#endif

  constexpr FakeComplex& operator+=(const FakeComplex& other)
  {
    real_ += other.real_;
    imag_ += other.imag_;
    return *this;
  }
  constexpr FakeComplex& operator-=(const FakeComplex& other)
  {
    real_ -= other.real_;
    imag_ -= other.imag_;
    return *this;
  }
  constexpr FakeComplex& operator*=(const FakeComplex& other)
  {
    real_ = real_ * other.real_ - imag_ * other.imag_;
    imag_ = imag_ * other.real_ + real_ * other.imag_;
    return *this;
  }
  constexpr FakeComplex& operator/=(const FakeComplex& other)
  {
    // just for illustration; please don't implement it this way.
    const auto other_mag = other.real_ * other.real_ + other.imag_ * other.imag_;
    real_ = (real_ * other.real_ + imag_ * other.imag_) / other_mag;
    imag_ = (imag_ * other.real_ - real_ * other.imag_) / other_mag;
    return *this;
  }

  constexpr FakeComplex& operator+=(const double other)
  {
    real_ += other;
    return *this;
  }
  constexpr FakeComplex& operator-=(const double other)
  {
    real_ -= other;
    return *this;
  }
  constexpr FakeComplex& operator*=(const double other)
  {
    real_ *= other;
    imag_ *= other;
    return *this;
  }
  constexpr FakeComplex& operator/=(const double other)
  {
    real_ /= other;
    imag_ /= other;
    return *this;
  }
};

// Unary operators

FakeComplex operator+( const FakeComplex& val )
{
  return val;
}
FakeComplex operator-( const FakeComplex& val )
{
  return {-real(val), -imag(val)};
}

// Binary homogeneous operators

FakeComplex operator+(const FakeComplex& z, const FakeComplex& w)
{
  return {real(z) + real(w), imag(z) + imag(w)};
}
FakeComplex operator-(const FakeComplex& z, const FakeComplex& w)
{
  return {real(z) - real(w), imag(z) - imag(w)};
}
FakeComplex operator*(const FakeComplex& z, const FakeComplex& w)
{
  return {real(z) * real(w) - imag(z) * imag(w),
    imag(z) * real(w) + real(z) * imag(w)};
}
FakeComplex operator/(const FakeComplex& z, const FakeComplex& w)
{
  // just for illustration; please don't implement it this way.
  const auto w_mag = real(w) * real(w) + imag(w) * imag(w);
  return {(real(z) * real(w) + imag(z) * imag(w)) / w_mag,
    (imag(z) * real(w) - real(z) * imag(w)) / w_mag};
}

// Binary (complex,real) operators

FakeComplex operator+(const FakeComplex& z, const double w)
{
  return {real(z) + w, imag(z)};
}
FakeComplex operator-(const FakeComplex& z, const double w)
{
  return {real(z) - w, imag(z)};
}
FakeComplex operator*(const FakeComplex& z, const double w)
{
  return {real(z) * w, imag(z) * w};
}
FakeComplex operator/(const FakeComplex& z, const double w)
{
  return {real(z) / w, imag(z) / w};
}

// Binary (real,complex) operators

FakeComplex operator+(const double z, const FakeComplex& w)
{
  return {z + real(w), z + imag(w)};
}
FakeComplex operator-(const double z, const FakeComplex& w)
{
  return {z - real(w), -imag(w)};
}
FakeComplex operator*(const double z, const FakeComplex& w)
{
  return {z * real(w), z * imag(w)};
}
FakeComplex operator/(const double z, const FakeComplex& w)
{
  // just for illustration; please don't implement it this way.
  const auto w_mag = real(w) * real(w) + imag(w) * imag(w);
  return {
    (z * real(w)) / w_mag,
    (-(z * imag(w))) / w_mag
  };
}

// Specialize test helper traits (P1673 does NOT need these)
namespace test_helpers {

template<class T>
static constexpr bool is_atomic_ref_not_arithmetic_v = false;

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
template<class U>
static constexpr bool is_atomic_ref_not_arithmetic_v<std::atomic_ref<U>> = ! std::is_arithmetic_v<U>;
#endif

} // namespace test_helpers

///////////////////////////////////////////////////////////
// conj_if_needed tests
///////////////////////////////////////////////////////////

template<class Real>
void test_real_conj_if_needed()
{
  using LinearAlgebra::impl::conj_if_needed;

  Real z(2.0);
  const Real z_conj_expected(2.0);

  auto z_conj = conj_if_needed(z);
  static_assert(std::is_same_v<decltype(z_conj), Real>);
  EXPECT_EQ(z_conj, z_conj_expected);
}

template<class Real>
void test_complex_conj_if_needed()
{
  using LinearAlgebra::impl::conj_if_needed;

  std::complex<Real> z(2.0, -3.0);
  const std::complex<Real> z_conj_expected(2.0, 3.0);

  auto z_conj = conj_if_needed(z);
  static_assert(std::is_same_v<decltype(z_conj), std::complex<Real>>);
  EXPECT_EQ(z_conj, z_conj_expected);
}

///////////////////////////////////////////////////////////
// conjugated_scalar tests
///////////////////////////////////////////////////////////

template<class Value>
Value get_test_xvalue(const Value&)
{
  static_assert(std::is_arithmetic_v<Value>);
  return Value(5);
}

template<class Real>
std::complex<Real> get_test_xvalue(const std::complex<Real>&)
{
    return std::complex<Real>(5.0, -6.0);
}

FakeComplex get_test_xvalue(const FakeComplex&)
{
    return {5.0, -6.0};
}

template<class Reference, class Value>
void test_conjugated_scalar_from_reference(Reference zd, Value zd_orig)
{
  using test_helpers::is_atomic_ref_not_arithmetic_v;
  using LinearAlgebra::impl::conj_if_needed;
  using LinearAlgebra::conjugated_scalar;
  using value_type = typename std::remove_cv_t<Value>;

#ifdef P1673_CONJUGATED_SCALAR_ARITHMETIC_OPERATORS_REFERENCE_OVERLOADS
  constexpr bool test_references = true;
#else
  constexpr bool test_references = is_atomic_ref_not_arithmetic_v<Reference>;
#endif

  std::cerr << "test_conjugated_scalar_from_reference" << std::endl;

  // Test conjugated_scalar constructor
  conjugated_scalar<Reference, value_type> cszd(zd);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {
      std::cerr << "Test conjugated_scalar + Reference" << std::endl;
      value_type left_add_result = cszd + zd;
      value_type left_add_result_expected = conj_if_needed(zd_orig) + zd_orig;
      EXPECT_EQ(left_add_result, left_add_result_expected);
      if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
        EXPECT_EQ(zd.load(), zd_orig);
      } else {
	EXPECT_EQ(zd, zd_orig);
      }

      std::cerr << "Test conjugated_scalar - Reference" << std::endl;
      value_type left_sub_result = cszd - zd;
      value_type left_sub_result_expected = conj_if_needed(zd_orig) - zd_orig;
      EXPECT_EQ(left_sub_result, left_sub_result_expected);
      if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
        EXPECT_EQ(zd.load(), zd_orig);
      } else {
        EXPECT_EQ(zd, zd_orig);
      }

      std::cerr << "Test conjugated_scalar * Reference" << std::endl;
      value_type left_mul_result = cszd * zd;
      value_type left_mul_result_expected = conj_if_needed(zd_orig) * zd_orig;
      EXPECT_EQ(left_mul_result, left_mul_result_expected);
      if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
        EXPECT_EQ(zd.load(), zd_orig);
      } else {
        EXPECT_EQ(zd, zd_orig);
      }

      std::cerr << "Test conjugated_scalar / Reference" << std::endl;
      value_type left_div_result = cszd / zd;
      value_type left_div_result_expected = conj_if_needed(zd_orig) / zd_orig;
      EXPECT_EQ(left_div_result, left_div_result_expected);
      if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
        EXPECT_EQ(zd.load(), zd_orig);
      } else {
        EXPECT_EQ(zd, zd_orig);
      }
    } // test_references

  std::cerr << "Test conjugated_scalar + value_type&&" << std::endl;
  value_type left_add_result2 = cszd + get_test_xvalue(value_type{});
  value_type left_add_result2_expected =
    conj_if_needed(zd_orig) + get_test_xvalue(value_type{});
  EXPECT_EQ(left_add_result2, left_add_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test conjugated_scalar - value_type&&" << std::endl;
  value_type left_sub_result2 = cszd - get_test_xvalue(value_type{});
  value_type left_sub_result2_expected =
    conj_if_needed(zd_orig) - get_test_xvalue(value_type{});
  EXPECT_EQ(left_sub_result2, left_sub_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test conjugated_scalar * value_type&&" << std::endl;
  value_type left_mul_result2 = cszd * get_test_xvalue(value_type{});
  value_type left_mul_result2_expected =
    conj_if_needed(zd_orig) * get_test_xvalue(value_type{});
  EXPECT_EQ(left_mul_result2, left_mul_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test conjugated_scalar / value_type&&" << std::endl;
  value_type left_div_result2 = cszd / get_test_xvalue(value_type{});
  value_type left_div_result2_expected =
    conj_if_needed(zd_orig) / get_test_xvalue(value_type{});
  EXPECT_EQ(left_div_result2, left_div_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {

    std::cerr << "Test Reference + conjugated_scalar" << std::endl;
    value_type right_add_result = zd + cszd;
    value_type right_add_result_expected = zd_orig + conj_if_needed(zd_orig);
    EXPECT_EQ(right_add_result, right_add_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    std::cerr << "Test Reference - conjugated_scalar" << std::endl;
    value_type right_sub_result = zd - cszd;
    value_type right_sub_result_expected = zd_orig - conj_if_needed(zd_orig);
    EXPECT_EQ(right_sub_result, right_sub_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    std::cerr << "Test Reference * conjugated_scalar" << std::endl;
    value_type right_mul_result = zd * cszd;
    value_type right_mul_result_expected = zd_orig * conj_if_needed(zd_orig);
    EXPECT_EQ(right_mul_result, right_mul_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    std::cerr << "Test Reference / conjugated_scalar" << std::endl;
    value_type right_div_result = zd / cszd;
    value_type right_div_result_expected = zd_orig / conj_if_needed(zd_orig);
    EXPECT_EQ(right_div_result, right_div_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

  } // test_references

  std::cerr << "Test value_type&& + conjugated_scalar" << std::endl;
  value_type right_add_result2 = get_test_xvalue(value_type{}) + cszd;
  value_type right_add_result2_expected =
    get_test_xvalue(value_type{}) + conj_if_needed(zd_orig);
  EXPECT_EQ(right_add_result2, right_add_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test value_type&& - conjugated_scalar" << std::endl;
  value_type right_sub_result2 = get_test_xvalue(value_type{}) - cszd;
  value_type right_sub_result2_expected =
    get_test_xvalue(value_type{}) - conj_if_needed(zd_orig);
  EXPECT_EQ(right_sub_result2, right_sub_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test value_type&& * conjugated_scalar" << std::endl;
  value_type right_mul_result2 = get_test_xvalue(value_type{}) * cszd;
  value_type right_mul_result2_expected =
    get_test_xvalue(value_type{}) * conj_if_needed(zd_orig);
  EXPECT_EQ(right_mul_result2, right_mul_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test value_type&& / conjugated_scalar" << std::endl;
  value_type right_div_result2 = get_test_xvalue(value_type{}) / cszd;
  value_type right_div_result2_expected =
    get_test_xvalue(value_type{}) / conj_if_needed(zd_orig);
  EXPECT_EQ(right_div_result2, right_div_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "Test that unary negate can be assigned to value_type" << std::endl;
  value_type unary_negate_result = -cszd;
  value_type unary_negate_result_expected = -conj_if_needed(zd_orig);
  EXPECT_EQ(unary_negate_result, unary_negate_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {

    std::cerr << "Test (unary negate) + Reference" << std::endl;
    value_type unary_negate_ref_result;
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      unary_negate_ref_result = -cszd + zd.load();
    } else {
      unary_negate_ref_result = -cszd + zd;
    }
    value_type unary_negate_ref_result_expected =
      -conj_if_needed(zd_orig) + zd_orig;
    EXPECT_EQ(unary_negate_ref_result, unary_negate_ref_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    std::cerr << "Test Reference + (unary negate)" << std::endl;
    value_type unary_negate_ref2_result;
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      unary_negate_ref2_result = zd.load() + (-cszd);
    } else {
      unary_negate_ref2_result = zd + (-cszd);
    }
    value_type unary_negate_ref2_result_expected =
      zd_orig + (-conj_if_needed(zd_orig));
    EXPECT_EQ(unary_negate_ref2_result, unary_negate_ref2_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

  } // test_references

  // Test (unary negate) + value_type&&
  value_type unary_negate_expr_result = -cszd + get_test_xvalue(value_type{});
  value_type unary_negate_expr_result_expected =
    -conj_if_needed(zd_orig) + get_test_xvalue(value_type{});
  EXPECT_EQ(unary_negate_expr_result, unary_negate_expr_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test value_type&& + (unary negate)
  value_type unary_negate_expr2_result =
    get_test_xvalue(value_type{}) + (-cszd);
  value_type unary_negate_expr2_result_expected =
    get_test_xvalue(value_type{}) + (-conj_if_needed(zd_orig));
  EXPECT_EQ(unary_negate_expr2_result, unary_negate_expr2_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic_v<Reference>) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test abs
  {
    auto abs_result = abs(cszd);
    auto abs_result_expected = [&]() {
      if constexpr (std::is_unsigned_v<value_type>) {
	return zd_orig;
      } else {
	return abs(conj_if_needed(zd_orig));
      }
    }();
    EXPECT_EQ(abs_result, abs_result_expected);
  }

  // Test conj
  {
    auto conj_result = conj(cszd);
    auto conj_result_expected = zd_orig;
    EXPECT_EQ(conj_result, conj_result_expected);
  }
}

template<class Real>
void test_complex_conjugated_scalar()
{
  std::cerr << "test_complex_conjugated_scalar" << std::endl;

  std::complex<Real> zd_orig{2.0, -3.0};
  std::complex<Real> zd{2.0, -3.0};
  test_conjugated_scalar_from_reference<
    std::complex<Real>&, std::complex<Real>>(zd, zd_orig);

  const std::complex<Real> zd2_orig{-1.0, 3.0};
  const std::complex<Real> zd2{-1.0, 3.0};
  test_conjugated_scalar_from_reference<
    const std::complex<Real>&, std::complex<Real>>(zd2, zd2_orig);

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
  const std::complex<Real> zd3_orig{-1.0, -2.0};
  std::complex<Real> zd3{-1.0, -2.0};
  test_conjugated_scalar_from_reference<
    std::atomic_ref<std::complex<Real>>, std::complex<Real>>
      (std::atomic_ref{zd3}, zd3_orig);
#endif

  // FIXME (mfh 2022/06/03) We might not need to worry about the comment below.
  //
  // float * atomic_ref<complex<float>> isn't a defined operator,
  // so unless we want to define it, we have to use a different reference type.
  {
    using value_type = std::complex<Real>;
    using inner_reference_type = value_type&;
    using LinearAlgebra::scaled_scalar;
    using reference_type = scaled_scalar<Real, inner_reference_type, value_type>;

    const Real scalingFactor = 3.0;
    const value_type zd4_orig = scalingFactor * value_type{-1.0, -2.0};
    value_type zd4{-1.0, -2.0};
    test_conjugated_scalar_from_reference<reference_type, value_type>(
								      reference_type(scalingFactor, inner_reference_type(zd4)),
								      zd4_orig);
  }
}

template<class Value>
void test_arithmetic_conjugated_scalar()
{
  static_assert(std::is_arithmetic_v<Value>);

  std::cerr << "test_arithmetic_conjugated_scalar" << std::endl;

  Value zd_orig{2};
  Value zd{2};
  test_conjugated_scalar_from_reference<
    Value&, Value>(zd, zd_orig);

  const Value zd2_orig{3};
  const Value zd2{3};
  test_conjugated_scalar_from_reference<
    const Value&, Value>(zd2, zd2_orig);

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
  const Value zd3_orig{4};
  Value zd3{4};
  test_conjugated_scalar_from_reference<
    std::atomic_ref<Value>, Value>(std::atomic_ref{zd3}, zd3_orig);
#endif
}

void test_FakeComplex_conjugated_scalar()
{
    std::cerr << "test_FakeComplex_conjugated_scalar" << std::endl;

    FakeComplex zd_orig{2.0, -3.0};
    FakeComplex zd{2.0, -3.0};
    test_conjugated_scalar_from_reference<
        FakeComplex&, FakeComplex>(zd, zd_orig);

    const FakeComplex zd2_orig{-1.0, 3.0};
    const FakeComplex zd2{-1.0, 3.0};
    test_conjugated_scalar_from_reference<
        const FakeComplex&, FakeComplex>(zd2, zd2_orig);

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
    const FakeComplex zd3_orig{-1.0, -2.0};
    FakeComplex zd3{-1.0, -2.0};
    test_conjugated_scalar_from_reference<
      std::atomic_ref<FakeComplex>, FakeComplex>(
          std::atomic_ref{zd3}, zd3_orig);
#endif
}

template<class ScalingFactor, class Reference, class Value>
void test_scaled_scalar_from_reference(
    ScalingFactor sf, Reference zd, Value zd_orig)
{
  std::cerr << "test_scaled_scalar_from_reference" << std::endl;

  using LinearAlgebra::impl::conj_if_needed;
  using LinearAlgebra::scaled_scalar;
  using value_type = typename std::remove_cv_t<Value>;
  constexpr bool is_atomic_ref_not_arithmetic =
    test_helpers::is_atomic_ref_not_arithmetic_v<Reference>;

#ifdef P1673_CONJUGATED_SCALAR_ARITHMETIC_OPERATORS_REFERENCE_OVERLOADS
  constexpr bool test_references = true;
#else
  constexpr bool test_references = is_atomic_ref_not_arithmetic;
#endif

  // Test scaled_scalar constructor
  scaled_scalar<ScalingFactor, Reference, Value> cszd(sf, zd);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {

    // Test scaled_scalar + Reference
    value_type left_add_result = cszd + zd;
    value_type left_add_result_expected = (sf * zd_orig) + zd_orig;
    EXPECT_EQ(left_add_result, left_add_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test scaled_scalar - Reference
    value_type left_sub_result = cszd - zd;
    value_type left_sub_result_expected = (sf * zd_orig) - zd_orig;
    EXPECT_EQ(left_sub_result, left_sub_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test scaled_scalar * Reference
    value_type left_mul_result = cszd * zd;
    value_type left_mul_result_expected = (sf * zd_orig) * zd_orig;
    EXPECT_EQ(left_mul_result, left_mul_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test scaled_scalar / Reference
    value_type left_div_result = cszd / zd;
    value_type left_div_result_expected = (sf * zd_orig) / zd_orig;
    EXPECT_EQ(left_div_result, left_div_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

  } // test_references

  // Test scaled_scalar + value_type&&
  value_type left_add_result2 = cszd + get_test_xvalue(value_type{});
  value_type left_add_result2_expected =
    (sf * zd_orig) + get_test_xvalue(value_type{});
  EXPECT_EQ(left_add_result2, left_add_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test scaled_scalar - value_type&&
  value_type left_sub_result2 = cszd - get_test_xvalue(value_type{});
  value_type left_sub_result2_expected =
    (sf * zd_orig) - get_test_xvalue(value_type{});
  EXPECT_EQ(left_sub_result2, left_sub_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test scaled_scalar * value_type&&
  value_type left_mul_result2 = cszd * get_test_xvalue(value_type{});
  value_type left_mul_result2_expected =
    (sf * zd_orig) * get_test_xvalue(value_type{});
  EXPECT_EQ(left_mul_result2, left_mul_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test scaled_scalar / value_type&&
  value_type left_div_result2 = cszd / get_test_xvalue(value_type{});
  value_type left_div_result2_expected =
    (sf * zd_orig) / get_test_xvalue(value_type{});
  EXPECT_EQ(left_div_result2, left_div_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {

    // Test Reference + scaled_scalar
    value_type right_add_result = zd + cszd;
    value_type right_add_result_expected = zd_orig + (sf * zd_orig);
    EXPECT_EQ(right_add_result, right_add_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test Reference - scaled_scalar
    value_type right_sub_result = zd - cszd;
    value_type right_sub_result_expected = zd_orig - (sf * zd_orig);
    EXPECT_EQ(right_sub_result, right_sub_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test Reference * scaled_scalar
    value_type right_mul_result = zd * cszd;
    value_type right_mul_result_expected = zd_orig * (sf * zd_orig);
    EXPECT_EQ(right_mul_result, right_mul_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test Reference / scaled_scalar
    value_type right_div_result = zd / cszd;
    value_type right_div_result_expected = zd_orig / (sf * zd_orig);
    EXPECT_EQ(right_div_result, right_div_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

  } // test_references

  // Test value_type&& + scaled_scalar
  value_type right_add_result2 = get_test_xvalue(value_type{}) + cszd;
  value_type right_add_result2_expected =
    get_test_xvalue(value_type{}) + (sf * zd_orig);
  EXPECT_EQ(right_add_result2, right_add_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test value_type&& - scaled_scalar
  value_type right_sub_result2 = get_test_xvalue(value_type{}) - cszd;
  value_type right_sub_result2_expected =
    get_test_xvalue(value_type{}) - (sf * zd_orig);
  EXPECT_EQ(right_sub_result2, right_sub_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test value_type&& * scaled_scalar
  value_type right_mul_result2 = get_test_xvalue(value_type{}) * cszd;
  value_type right_mul_result2_expected =
    get_test_xvalue(value_type{}) * (sf * zd_orig);
  EXPECT_EQ(right_mul_result2, right_mul_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test value_type&& / scaled_scalar
  value_type right_div_result2 = get_test_xvalue(value_type{}) / cszd;
  value_type right_div_result2_expected =
    get_test_xvalue(value_type{}) / (sf * zd_orig);
  EXPECT_EQ(right_div_result2, right_div_result2_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test that unary negate can be assigned to value_type.
  value_type unary_negate_result = -cszd;
  value_type unary_negate_result_expected = -(sf * zd_orig);
  EXPECT_EQ(unary_negate_result, unary_negate_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  if constexpr (test_references) {

    // Test (unary negate) + Reference
    value_type unary_negate_ref_result = -cszd + zd;
    value_type unary_negate_ref_result_expected =
      -(sf * zd_orig) + zd_orig;
    EXPECT_EQ(unary_negate_ref_result, unary_negate_ref_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

    // Test Reference + (unary negate)
    value_type unary_negate_ref2_result = zd + (-cszd);
    value_type unary_negate_ref2_result_expected =
      zd_orig + (-(sf * zd_orig));
    EXPECT_EQ(unary_negate_ref2_result, unary_negate_ref2_result_expected);
    if constexpr (is_atomic_ref_not_arithmetic) {
      EXPECT_EQ(zd.load(), zd_orig);
    } else {
      EXPECT_EQ(zd, zd_orig);
    }

  } // test_references

  // Test (unary negate) + value_type&&
  value_type unary_negate_expr_result = -cszd + get_test_xvalue(value_type{});
  value_type unary_negate_expr_result_expected =
    -(sf * zd_orig) + get_test_xvalue(value_type{});
  EXPECT_EQ(unary_negate_expr_result, unary_negate_expr_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test value_type&& + (unary negate)
  value_type unary_negate_expr2_result =
    get_test_xvalue(value_type{}) + (-cszd);
  value_type unary_negate_expr2_result_expected =
    get_test_xvalue(value_type{}) + (-(sf * zd_orig));
  EXPECT_EQ(unary_negate_expr2_result, unary_negate_expr2_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  // Test abs
  {
    auto abs_result = abs(cszd);
    auto abs_result_expected = [&]() {
      // Unsigned integers don't work nicely with abs.
      if constexpr (std::is_unsigned_v<value_type>) {
        return sf * zd;
      } else {
        return abs(sf * zd);
      }
    }();
    EXPECT_EQ(abs_result, abs_result_expected);
  }

  // Test conj
  {
    auto conj_result = conj(cszd);
    auto conj_result_expected = conj_if_needed(sf * zd);
    EXPECT_EQ(conj_result, conj_result_expected);
  }
}

template<class ScalingFactor, class Reference, class Value>
void test_two_scaled_scalars_from_reference(
  ScalingFactor sf, Reference zd, Value zd_orig,
  const char scalingFactorName[],
  const char referenceName[],
  const char valueName[])
{
  std::cerr << "test_two_scaled_scalars_from_reference<"
	    << scalingFactorName << ", " << referenceName
	    << ", " << valueName << ">" << std::endl;

  using LinearAlgebra::scaled_scalar;
  using value_type = typename std::remove_cv_t<Value>;
  constexpr bool is_atomic_ref_not_arithmetic =
    test_helpers::is_atomic_ref_not_arithmetic_v<Reference>;

  // Test scaled_scalar constructor
  scaled_scalar<ScalingFactor, Reference, Value> cszd1(sf, zd);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  scaled_scalar<ScalingFactor, Reference, Value> cszd2(sf, zd);

  std::cerr << "- Test scaled_scalar + scaled_scalar" << std::endl;
  value_type left_add_result = cszd1 + cszd2;
  value_type left_add_result_expected = (sf * zd_orig) + (sf * zd_orig);
  EXPECT_EQ(left_add_result, left_add_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "- Test scaled_scalar - scaled_scalar" << std::endl;
  value_type left_sub_result = cszd1 - cszd2;
  value_type left_sub_result_expected = (sf * zd_orig) - (sf * zd_orig);
  EXPECT_EQ(left_sub_result, left_sub_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "- Test scaled_scalar * scaled_scalar" << std::endl;
  value_type left_mul_result = cszd1 * cszd2;
  value_type left_mul_result_expected = (sf * zd_orig) * (sf * zd_orig);
  EXPECT_EQ(left_mul_result, left_mul_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }

  std::cerr << "- Test scaled_scalar / scaled_scalar" << std::endl;
  value_type left_div_result = cszd1 / cszd2;
  value_type left_div_result_expected = (sf * zd_orig) / (sf * zd_orig);
  EXPECT_EQ(left_div_result, left_div_result_expected);
  if constexpr (is_atomic_ref_not_arithmetic) {
    EXPECT_EQ(zd.load(), zd_orig);
  } else {
    EXPECT_EQ(zd, zd_orig);
  }
}

template<class ScalingFactor, class Real>
void test_complex_scaled_scalar(const ScalingFactor& scalingFactor,
    const char scalingFactorName[], const char realName[])
{
    std::cerr << "test_complex_scaled_scalar" << std::endl;

    std::complex<Real> zd_orig{2.0, -3.0};
    std::complex<Real> zd{2.0, -3.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, std::complex<Real>&, std::complex<Real>>(scalingFactor, zd, zd_orig);

    const std::complex<Real> zd2_orig{-1.0, 3.0};
    const std::complex<Real> zd2{-1.0, 3.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, const std::complex<Real>&, std::complex<Real>>(
            scalingFactor, zd2, zd2_orig);

    // NOTE (mfh 2022/06/03) This doesn't compile, but that's probably
    // a result of atomic_ref<complex<R>> not having arithmetic
    // operators, not a result of our design.
#if 0
#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
    const std::complex<Real> zd3_orig{-1.0, -2.0};
    std::complex<Real> zd3{-1.0, -2.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, std::atomic_ref<std::complex<Real>>, std::complex<Real>>(
            scalingFactor, std::atomic_ref{zd3}, zd3_orig);
#endif
#endif // 0

    const std::string valueName = std::string("std::complex<") + realName + ">";
    const std::string valueRefName = std::string("const ") + valueName + "&";
    test_two_scaled_scalars_from_reference<
        ScalingFactor, const std::complex<Real>&, std::complex<Real>>(
            scalingFactor, zd2, zd2_orig,
            scalingFactorName, valueRefName.c_str(), valueName.c_str());
}

template<class Value>
void test_arithmetic_scaled_scalar(const char valueName[])
{
  static_assert(std::is_arithmetic_v<Value>);

  std::cerr << "test_arithmetic_scaled_scalar" << std::endl;

  const Value scalingFactor(3);
  Value zd_orig{2};
  Value zd{2};
  test_scaled_scalar_from_reference<
    Value, Value&, Value>(scalingFactor, zd, zd_orig);

  const Value zd2_orig{3};
  const Value zd2{3};
  test_scaled_scalar_from_reference<
    Value, const Value&, Value>(scalingFactor, zd2, zd2_orig);

  const std::string valueRefName = std::string("const ") + valueName + "&";
  test_two_scaled_scalars_from_reference<
    Value, const Value&, Value>(scalingFactor, zd2, zd2_orig,
				valueName, valueRefName.c_str(), valueName);
}

template<class ScalingFactor>
void test_FakeComplex_scaled_scalar(const ScalingFactor& scalingFactor)
{
    std::cerr << "test_FakeComplex_scaled_scalar" << std::endl;

    FakeComplex zd_orig{2.0, -3.0};
    FakeComplex zd{2.0, -3.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, FakeComplex&, FakeComplex>(
            scalingFactor, zd, zd_orig);

    const FakeComplex zd2_orig{-1.0, 3.0};
    const FakeComplex zd2{-1.0, 3.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, const FakeComplex&, FakeComplex>(
            scalingFactor, zd2, zd2_orig);
}

namespace {
  TEST(proxy_refs, conj_if_needed)
  {
    test_complex_conj_if_needed<float>();
    test_complex_conj_if_needed<double>();
    test_complex_conj_if_needed<long double>();

    test_real_conj_if_needed<float>();
    test_real_conj_if_needed<double>();
    test_real_conj_if_needed<long double>();

    test_real_conj_if_needed<int32_t>();
    test_real_conj_if_needed<uint32_t>();
    test_real_conj_if_needed<int64_t>();
    test_real_conj_if_needed<uint64_t>();
  }

  template<class R>
  void test_imag_if_needed_complex()
  {
    using LinearAlgebra::impl::imag_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = imag_if_needed(z);
    EXPECT_EQ(z_imag, R(4.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_imag_if_needed_floating_point()
  {
    using LinearAlgebra::impl::imag_if_needed;
    T x = 9.0;
    auto x_imag = imag_if_needed(x);
    EXPECT_EQ(x_imag, T(0.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_imag_if_needed_integral()
  {
    using LinearAlgebra::impl::imag_if_needed;
    T x = 3;
    auto x_imag = imag_if_needed(x);
    EXPECT_EQ(x_imag, T(0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }

  TEST(proxy_refs, imag_if_needed)
  {
    test_imag_if_needed_complex<float>();
    test_imag_if_needed_complex<double>();
    test_imag_if_needed_complex<long double>();

    test_imag_if_needed_floating_point<float>();
    test_imag_if_needed_floating_point<double>();
    test_imag_if_needed_floating_point<long double>();

    test_imag_if_needed_integral<int8_t>();
    test_imag_if_needed_integral<uint8_t>();
    test_imag_if_needed_integral<int16_t>();
    test_imag_if_needed_integral<uint16_t>();
    test_imag_if_needed_integral<int32_t>();
    test_imag_if_needed_integral<uint32_t>();
    test_imag_if_needed_integral<int64_t>();
    test_imag_if_needed_integral<uint64_t>();

    {
      using LinearAlgebra::impl::imag_if_needed;
      FakeComplex z{3.0, 4.0};
      auto z_imag = imag_if_needed(z);
      EXPECT_EQ(z_imag, 4.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(imag(z))>);
    }
    {
      using LinearAlgebra::impl::imag_if_needed;
      FakeRealNumber x{3.0};
      auto x_imag = imag_if_needed(x);
      EXPECT_EQ(x_imag, FakeRealNumber{});
      static_assert(std::is_same_v<decltype(x_imag), FakeRealNumber>);
    }
  }

  template<class R>
  void test_real_if_needed_complex()
  {
    using LinearAlgebra::impl::real_if_needed;
    std::complex<R> z{R(3.0), R(4.0)};
    auto z_imag = real_if_needed(z);
    EXPECT_EQ(z_imag, R(3.0));
    static_assert(std::is_same_v<decltype(z_imag), R>);
  }
  template<class T>
  void test_real_if_needed_floating_point()
  {
    using LinearAlgebra::impl::real_if_needed;
    T x = 9.0;
    auto x_imag = real_if_needed(x);
    EXPECT_EQ(x_imag, T(9.0));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }
  template<class T>
  void test_real_if_needed_integral()
  {
    using LinearAlgebra::impl::real_if_needed;
    T x = 3;
    auto x_imag = real_if_needed(x);
    EXPECT_EQ(x_imag, T(3));
    static_assert(std::is_same_v<decltype(x_imag), T>);
  }

  TEST(proxy_refs, real_if_needed)
  {
    test_real_if_needed_complex<float>();
    test_real_if_needed_complex<double>();
    test_real_if_needed_complex<long double>();

    test_real_if_needed_floating_point<float>();
    test_real_if_needed_floating_point<double>();
    test_real_if_needed_floating_point<long double>();

    test_real_if_needed_integral<int8_t>();
    test_real_if_needed_integral<uint8_t>();
    test_real_if_needed_integral<int16_t>();
    test_real_if_needed_integral<uint16_t>();
    test_real_if_needed_integral<int32_t>();
    test_real_if_needed_integral<uint32_t>();
    test_real_if_needed_integral<int64_t>();
    test_real_if_needed_integral<uint64_t>();

    {
      using LinearAlgebra::impl::real_if_needed;
      FakeComplex z{3.0, 4.0};
      auto z_imag = real_if_needed(z);
      EXPECT_EQ(z_imag, 3.0);
      static_assert(std::is_same_v<decltype(z_imag), decltype(imag(z))>);
    }
    {
      using LinearAlgebra::impl::real_if_needed;
      FakeRealNumber x{3.0};
      auto x_real = real_if_needed(x);
      EXPECT_EQ(x_real, FakeRealNumber{3.0});
      static_assert(std::is_same_v<decltype(x_real), FakeRealNumber>);
    }
  }

  TEST(proxy_refs, conjugated_scalar)
  {
    test_complex_conjugated_scalar<float>();
    test_complex_conjugated_scalar<double>();
    test_complex_conjugated_scalar<long double>();

    test_arithmetic_conjugated_scalar<float>();
    test_arithmetic_conjugated_scalar<double>();
    test_arithmetic_conjugated_scalar<long double>();

    test_arithmetic_conjugated_scalar<int32_t>();
    test_arithmetic_conjugated_scalar<uint32_t>();
    test_arithmetic_conjugated_scalar<int64_t>();
    test_arithmetic_conjugated_scalar<uint64_t>();

    test_FakeComplex_conjugated_scalar();

    FakeRealNumber fn{4.2};
    using LinearAlgebra::conjugated_scalar;
    conjugated_scalar<FakeRealNumber&, FakeRealNumber> fncs(fn);
    EXPECT_EQ(fn, FakeRealNumber(fncs));
  }

  TEST(proxy_refs, scaled_scalar)
  {
    test_complex_scaled_scalar<float, float>(float(4.0), "float", "float");
    test_complex_scaled_scalar<double, double>(double(4.0), "double", "double");
    {
        using scaling_factor_type = long double;
        test_complex_scaled_scalar<scaling_factor_type, long double>(
            scaling_factor_type(4.0), "long double", "long double");
    }
    {
        using scaling_factor_type = std::complex<float>;
        test_complex_scaled_scalar<scaling_factor_type, float>(
            scaling_factor_type(4.0, 5.0), "std::complex<float>", "float");
    }
    {
        using scaling_factor_type = std::complex<double>;
        test_complex_scaled_scalar<scaling_factor_type, double>(
            scaling_factor_type(4.0, 5.0), "std::complex<double>", "double");
    }
    {
        using scaling_factor_type = std::complex<long double>;
        test_complex_scaled_scalar<scaling_factor_type, long double>(
            scaling_factor_type(4.0, 5.0), "std::complex<long double>", "long double");
    }

    test_arithmetic_scaled_scalar<float>("float");
    test_arithmetic_scaled_scalar<double>("double");
    test_arithmetic_scaled_scalar<long double>("long double");

    test_arithmetic_scaled_scalar<int32_t>("int32_t");
    test_arithmetic_scaled_scalar<uint32_t>("uint32_t");
    test_arithmetic_scaled_scalar<int64_t>("int64_t");
    test_arithmetic_scaled_scalar<uint64_t>("uint64_t");

    test_FakeComplex_scaled_scalar(double(4.0));
    test_FakeComplex_scaled_scalar(FakeComplex{4.0, 5.0});
  }
} // namespace (anonymous)
