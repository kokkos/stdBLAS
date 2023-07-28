/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TEST_NUMBERS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TEST_NUMBERS_HPP_

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

// conj_if_needed assumes that FakeRealNumber is a real number,
// because it doesn't have an ADL-accessible definition of conj.
FakeRealNumber conj(const FakeRealNumber& x) { return x; }

///////////////////////////////////////////////////////////
// Custom real number type for tests
///////////////////////////////////////////////////////////

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

FakeComplex operator+(const FakeComplex& val)
{
  return val;
}
FakeComplex operator-(const FakeComplex& val)
{
  return { -val.real, -val.imag };
}

// Binary homogeneous operators

FakeComplex operator+(const FakeComplex& z, const FakeComplex& w)
{
  return { z.real + w.real, z.imag + w.imag };
}
FakeComplex operator-(const FakeComplex& z, const FakeComplex& w)
{
  return { z.real - w.real, z.imag - w.imag };
}
FakeComplex operator*(const FakeComplex& z, const FakeComplex& w)
{
  return { z.real * w.real - z.imag * w.imag,
      z.imag * w.real + z.real * w.imag };
}
FakeComplex operator/(const FakeComplex& z, const FakeComplex& w)
{
  // just for illustration; please don't implement it this way.
  const auto w_mag = w.real * w.real + w.imag * w.imag;
  return { (z.real * w.real + z.imag * w.imag) / w_mag,
      (z.imag * w.real - z.real * w.imag) / w_mag };
}

// Binary (complex,real) operators

FakeComplex operator+(const FakeComplex& z, const double w)
{
  return { z.real + w, z.imag };
}
FakeComplex operator-(const FakeComplex& z, const double w)
{
  return { z.real - w, z.imag };
}
FakeComplex operator*(const FakeComplex& z, const double w)
{
  return { z.real * w, z.imag * w };
}
FakeComplex operator/(const FakeComplex& z, const double w)
{
  return { z.real / w, z.imag / w };
}

// Binary (real,complex) operators

FakeComplex operator+(const double z, const FakeComplex& w)
{
  return { z + w.real, z + w.imag };
}
FakeComplex operator-(const double z, const FakeComplex& w)
{
  return { z - w.real, -w.imag };
}
FakeComplex operator*(const double z, const FakeComplex& w)
{
  return { z * w.real, z * w.imag };
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

// conj_if_needed knows that FakeComplex is a complex number
// because it has this ADL-findable conj function.
// Ditto for abs, real, and imag below.
FakeComplex conj(const FakeComplex& z) { return { z.real, -z.imag }; }

auto abs(const FakeComplex& z) { return sqrt(z.real * z.real + z.imag * z.imag); }

auto real(const FakeComplex& z) { return z.real; }

auto imag(const FakeComplex& z) { return z.imag; }

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TEST_NUMBERS_HPP_
