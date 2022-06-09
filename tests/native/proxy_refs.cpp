// To try using subscript operator comment in macro below
// the header will by default also check for the feature macro, and enable it
// defining the macro to 0 will overwrite the automatic setting
// x86-64 clang (experimental auto NSDMI) supports the operator, but you need
// to explicitly comment in below macro
//#define MDSPAN_USE_BRACKET_OPERATOR 1

// To force enable operator() comment in the macro below
// You can enable both at the same time. 
//#define MDSPAN_USE_PAREN_OPERATOR 0

#define P1673_CONJUGATED_SCALAR_ARITHMETIC_OPERATORS_REFERENCE_OVERLOADS 1

#include "gtest/gtest.h"
#include <experimental/linalg>

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
// Non-arithmetic types need a definition of conj,
// else conjugated_scalar won't compile with them.
FakeRealNumber conj(const FakeRealNumber& x) { return x; }

// Custom complex number type

struct FakeComplex {
    double real;
    double imag;

#ifdef __cpp_impl_three_way_comparison
    bool operator==(const FakeComplex&) const = default;
#else
  friend bool operator==(const FakeComplex& x, const FakeComplex& y)
  {
    return x.real == y.real && x.imag == y.imag;
  }
#endif

    constexpr FakeComplex& operator+=(const FakeComplex& other)
    {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    constexpr FakeComplex& operator-=(const FakeComplex& other)
    {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }
    constexpr FakeComplex& operator*=(const FakeComplex& other)
    {
        real = real * other.real - imag * other.imag;
        imag = imag * other.real + real * other.imag;
        return *this;
    }
    constexpr FakeComplex& operator/=(const FakeComplex& other)
    {
        // just for illustration; please don't implement it this way.
        const auto other_mag = other.real * other.real + other.imag * other.imag;
        real = (real * other.real + imag * other.imag) / other_mag;
        imag = (imag * other.real - real * other.imag) / other_mag;
        return *this;
    }

    constexpr FakeComplex& operator+=(const double other)
    {
        real += other;
        return *this;
    }
    constexpr FakeComplex& operator-=(const double other)
    {
        real -= other;
        return *this;
    }
    constexpr FakeComplex& operator*=(const double other)
    {
        real *= other;
        imag *= other;
        return *this;
    }
    constexpr FakeComplex& operator/=(const double other)
    {
        real /= other;
        imag /= other;
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
    return {-val.real, -val.imag};
}

// Binary homogeneous operators

FakeComplex operator+(const FakeComplex& z, const FakeComplex& w)
{
    return {z.real + w.real, z.imag + w.imag};
}
FakeComplex operator-(const FakeComplex& z, const FakeComplex& w)
{
    return {z.real - w.real, z.imag - w.imag};
}
FakeComplex operator*(const FakeComplex& z, const FakeComplex& w)
{
    return {z.real * w.real - z.imag * w.imag,
        z.imag * w.real + z.real * w.imag};
}
FakeComplex operator/(const FakeComplex& z, const FakeComplex& w)
{
    // just for illustration; please don't implement it this way.
    const auto w_mag = w.real * w.real + w.imag * w.imag;
    return {(z.real * w.real + z.imag * w.imag) / w_mag,
        (z.imag * w.real - z.real * w.imag) / w_mag};
}

// Binary (complex,real) operators

FakeComplex operator+(const FakeComplex& z, const double w)
{
    return {z.real + w, z.imag};
}
FakeComplex operator-(const FakeComplex& z, const double w)
{
    return {z.real - w, z.imag};
}
FakeComplex operator*(const FakeComplex& z, const double w)
{
    return {z.real * w, z.imag * w};
}
FakeComplex operator/(const FakeComplex& z, const double w)
{
    return {z.real / w, z.imag / w};
}

// Binary (real,complex) operators

FakeComplex operator+(const double z, const FakeComplex& w)
{
    return {z + w.real, z + w.imag};
}
FakeComplex operator-(const double z, const FakeComplex& w)
{
    return {z - w.real, -w.imag};
}
FakeComplex operator*(const double z, const FakeComplex& w)
{
    return {z * w.real, z * w.imag};
}
FakeComplex operator/(const double z, const FakeComplex& w)
{
    // just for illustration; please don't implement it this way.
    const auto w_mag = w.real * w.real + w.imag * w.imag;
    return {
        (z * w.real) / w_mag,
        (-(z * w.imag)) / w_mag
    };
}

// Specialize test helper traits (P1673 does NOT need these)
namespace test_helpers {

template<class T>
constexpr bool is_complex_v = false;

template<>
constexpr bool is_complex_v<std::complex<float>> = true;

template<>
constexpr bool is_complex_v<std::complex<double>> = true;

template<>
constexpr bool is_complex_v<std::complex<long double>> = true;

template<>
constexpr bool is_complex_v<FakeComplex> = true;

template<class T>
static constexpr bool is_atomic_ref_not_arithmetic_v = false;

#ifdef __cpp_lib_atomic_ref
template<class U>
static constexpr bool is_atomic_ref_not_arithmetic_v<std::atomic_ref<U>> = ! std::is_arithmetic_v<U>;
#endif // __cpp_lib_atomic_ref

} // namespace test_helpers

// FakeComplex conj implementation

FakeComplex conj(const FakeComplex& z) { return {z.real, -z.imag}; }

///////////////////////////////////////////////////////////
// conj_if_needed tests
///////////////////////////////////////////////////////////

template<class Real>
void test_real_conj_if_needed()
{
  using std::experimental::linalg::impl::conj_if_needed;

  Real z(2.0);
  const Real z_conj_expected(2.0);

  auto z_conj = conj_if_needed(z);
  static_assert(std::is_same_v<decltype(z_conj), Real>);
  EXPECT_EQ(z_conj, z_conj_expected);
}

template<class Real>
void test_complex_conj_if_needed()
{
  using std::experimental::linalg::impl::conj_if_needed;
  
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
  using std::experimental::linalg::impl::conj_if_needed;
  using std::experimental::linalg::conjugated_scalar;  
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

#ifdef __cpp_lib_atomic_ref
  const std::complex<Real> zd3_orig{-1.0, -2.0};
  std::complex<Real> zd3{-1.0, -2.0};
  test_conjugated_scalar_from_reference<
    std::atomic_ref<std::complex<Real>>, std::complex<Real>>
      (std::atomic_ref{zd3}, zd3_orig);
#endif // __cpp_lib_atomic_ref

  // FIXME (mfh 2022/06/03) We might not need to worry about the comment below.
  //
  // float * atomic_ref<complex<float>> isn't a defined operator,
  // so unless we want to define it, we have to use a different reference type.
  {
    using value_type = std::complex<Real>;
    using inner_reference_type = value_type&;
    using std::experimental::linalg::scaled_scalar;
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

#ifdef __cpp_lib_atomic_ref
  const Value zd3_orig{4};
  Value zd3{4};
  test_conjugated_scalar_from_reference<
    std::atomic_ref<Value>, Value>(std::atomic_ref{zd3}, zd3_orig);
#endif // __cpp_lib_atomic_ref    
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

#ifdef __cpp_lib_atomic_ref    
    const FakeComplex zd3_orig{-1.0, -2.0};
    FakeComplex zd3{-1.0, -2.0};
    test_conjugated_scalar_from_reference<
      std::atomic_ref<FakeComplex>, FakeComplex>(
          std::atomic_ref{zd3}, zd3_orig);
#endif // __cpp_lib_atomic_ref
}

template<class ScalingFactor, class Reference, class Value>
void test_scaled_scalar_from_reference(
    ScalingFactor sf, Reference zd, Value zd_orig)
{
  std::cerr << "test_scaled_scalar_from_reference" << std::endl;

  using std::experimental::linalg::scaled_scalar;
  using value_type = typename std::remove_cv_t<Value>;
  constexpr bool is_atomic_ref_not_arithmetic = test_helpers::is_atomic_ref_not_arithmetic_v<Reference>;

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

  using std::experimental::linalg::scaled_scalar;  
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
#ifdef __cpp_lib_atomic_ref
    const std::complex<Real> zd3_orig{-1.0, -2.0};
    std::complex<Real> zd3{-1.0, -2.0};
    test_scaled_scalar_from_reference<
        ScalingFactor, std::atomic_ref<std::complex<Real>>, std::complex<Real>>(
            scalingFactor, std::atomic_ref{zd3}, zd3_orig);
#endif // __cpp_lib_atomic_ref
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

    FakeRealNumber fn;
    using std::experimental::linalg::conjugated_scalar;
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
