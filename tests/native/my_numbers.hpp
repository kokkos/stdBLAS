#ifndef LINALG_TESTS_NATIVE_MY_NUMBERS_HPP
#define LINALG_TESTS_NATIVE_MY_NUMBERS_HPP

#include "./gtest_fixtures.hpp"

namespace TestLinearAlgebra {

class MyReal {
public:
  MyReal() = default;
  explicit MyReal(double value) : value_(value) {}
  double value() const { return value_; }

  friend MyReal abs(MyReal x) { return MyReal{std::abs(x.value())}; }

  friend bool operator==(MyReal x, MyReal y) {
    return x.value() == y.value();
  }

private:
  double value_ = 0.0;
};

class MyComplex {
private:
  double real_ = 0.0;
  double imag_ = 0.0;

public:
  MyComplex() = default;
  MyComplex(double re, double im) : real_(re), imag_(im) {}

  friend double real(MyComplex z) {
    return z.real_;
  }

  friend double imag(MyComplex z) {
    return z.imag_;
  }

  friend double abs(MyComplex z) {
    return std::sqrt(z.real_ * z.real_ + z.imag_ * z.imag_);
  }

  friend MyComplex conj(MyComplex z) {
    return {z.real_, -z.imag_};
  }

  std::complex<double> value() const {
    return {real_, imag_};
  }

  friend bool operator==(MyComplex x, MyComplex y) {
    return x.value() == y.value();
  }
};

} // namespace TestLinearAlgebra

#endif // LINALG_TESTS_NATIVE_MY_NUMBERS_HPP
