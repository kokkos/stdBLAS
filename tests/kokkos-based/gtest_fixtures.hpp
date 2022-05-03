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

#ifndef LINALG_TESTS_KOKKOS_BLAS1_FIXTURES_HPP_
#define LINALG_TESTS_KOKKOS_BLAS1_FIXTURES_HPP_

#include <experimental/linalg>
#include <experimental/mdspan>
#include <Kokkos_Core.hpp>
#include "gtest/gtest.h"
#include <random>

// it is fine to put these here even if this
// is a header since this is limited to tests
using std::experimental::mdspan;
using std::experimental::extents;
using std::experimental::dynamic_extent;

//
// helper class for generating random numbers
//
template <class ValueType> struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::random_device rd;
  std::mt19937 m_gen{rd()};
  dist_type m_dist;

  UnifDist(const int a, const int b) : m_dist(a, b){}
  int operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::random_device rd;
  std::mt19937 m_gen{rd()};
  dist_type m_dist;

  UnifDist(const double a, const double b) : m_dist(a, b){}
  double operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<float> {
  using dist_type = std::uniform_real_distribution<float>;
  std::random_device rd;
  std::mt19937 m_gen{rd()};
  dist_type m_dist;

  UnifDist(const float a, const float b) : m_dist(a, b){}
  float operator()() { return m_dist(m_gen); }
};


template<class T, class mdspan_t>
void fill_random_mdspan(UnifDist<T> & randObj_r,
			UnifDist<T> & randObj_i,
			mdspan_t mdspanObj)
{
  static_assert( mdspan_t::rank() <=2 );

  if constexpr( mdspan_t::rank() == 1){
    for (std::size_t i=0; i<mdspanObj.extent(0); ++i) {
      mdspanObj(i) = {randObj_r(), randObj_i()};
    }
  }
  else{
    for (std::size_t i=0; i<mdspanObj.extent(0); ++i) {
      for (std::size_t j=0; j<mdspanObj.extent(1); ++j) {
	mdspanObj(i,j) = {randObj_r(), randObj_i()};
      }
    }
  }
}

template<class T, class mdspan_t>
void fill_random_mdspan(UnifDist<T> & randObj,
			mdspan_t mdspanObj)
{
  static_assert( mdspan_t::rank() <=2 );

  if constexpr( mdspan_t::rank() == 1){
    for (std::size_t i=0; i<mdspanObj.extent(0); ++i) {
      mdspanObj(i) = randObj();
    }
  }
  else{
    for (std::size_t i=0; i<mdspanObj.extent(0); ++i) {
      for (std::size_t j=0; j<mdspanObj.extent(1); ++j) {
	mdspanObj(i,j) = randObj();
      }
    }
  }
}


template<class T>
constexpr void static_check_value_type(T /*unused*/)
{
  static_assert(std::is_same_v<T, int> ||
		std::is_same_v<T, double> ||
		std::is_same_v<T, float> ||
		std::is_same_v<T, std::complex<double>>,
		"gtest_fixtures: unsupported value_type");
}


template<class T>
class _blas1_signed_fixture : public ::testing::Test
{
  // extent is arbitrarily chosen but not trivially small
  const std::size_t myExtent = 137;

public:
  using value_type = T;

  _blas1_signed_fixture()
    : x_view("x_view", myExtent),
      y_view("y_view", myExtent),
      z_view("z_view", myExtent),
      x(x_view.data(), myExtent),
      y(y_view.data(), myExtent),
      z(z_view.data(), myExtent)
  {
    static_check_value_type(value_type{});

    if constexpr(std::is_same_v<T, std::complex<double>>){
      const auto a_r = static_cast<double>(-101);
      const auto b_r = static_cast<double>( 103);
      UnifDist<double> randObj_r(a_r, b_r);

      const auto a_i = static_cast<double>(-21);
      const auto b_i = static_cast<double>( 43);
      UnifDist<double> randObj_i(a_i, b_i);

      fill_random_mdspan(randObj_r, randObj_i, x_view);
      fill_random_mdspan(randObj_r, randObj_i, y_view);
      fill_random_mdspan(randObj_r, randObj_i, z_view);
    }
    else{
      const auto a = static_cast<value_type>(-11);
      const auto b = static_cast<value_type>( 23);
      UnifDist<value_type> randObj(a, b);

      fill_random_mdspan(randObj, x_view);
      fill_random_mdspan(randObj, y_view);
      fill_random_mdspan(randObj, z_view);
    }
  }

  // these views will be on default memory space
  Kokkos::View<value_type*, Kokkos::HostSpace> x_view;
  Kokkos::View<value_type*, Kokkos::HostSpace> y_view;
  Kokkos::View<value_type*, Kokkos::HostSpace> z_view;

  using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
  mdspan_t x;
  mdspan_t y;
  mdspan_t z;
};

template<class T>
class _blas2_signed_fixture : public ::testing::Test
{
protected:
  // extents are arbitrarily chosen but not trivially small
  const std::size_t myExtent0 = 77;
  const std::size_t myExtent1 = 41;

public:
  using value_type = T;

  _blas2_signed_fixture()
    : A_e0e1_view("A_e0e1_view", myExtent0, myExtent1),
      A_e0e1(A_e0e1_view.data(), myExtent0, myExtent1),
      //
      B_e0e1_view("B_e0e1_view", myExtent0, myExtent1),
      B_e0e1(B_e0e1_view.data(), myExtent0, myExtent1),
      //
      A_sym_e0_view("A_sym_e0_view", myExtent0, myExtent0),
      A_sym_e0(A_sym_e0_view.data(), myExtent0, myExtent0),
      //
      A_hem_e0_view("A_hem_e0_view", myExtent0, myExtent0),
      A_hem_e0(A_hem_e0_view.data(), myExtent0, myExtent0),
      //
      x_e0_view("x_e0_view", myExtent0),
      x_e0(x_e0_view.data(), myExtent0),
      //
      x_e1_view("x_e1_view", myExtent1),
      x_e1(x_e1_view.data(), myExtent1),
      //
      y_e0_view("y_e0_view", myExtent0),
      y_e0(y_e0_view.data(), myExtent0),
      //
      z_e0_view("z_e0_view", myExtent0),
      z_e0(z_e0_view.data(), myExtent0)
  {

    static_check_value_type(value_type{});

    if constexpr(std::is_same_v<T, std::complex<double>>){
      const auto a_r = static_cast<double>(-5);
      const auto b_r = static_cast<double>( 3);
      UnifDist<double> randObj_r(a_r, b_r);

      const auto a_i = static_cast<double>(-9);
      const auto b_i = static_cast<double>( 7);
      UnifDist<double> randObj_i(a_i, b_i);

      // fill symmetric matrices
      for (std::size_t i=0; i < myExtent0; ++i) {
	for (std::size_t j=i; j < myExtent0; ++j) {
	  A_sym_e0(i,j) = {randObj_r(), randObj_i()};
	  A_sym_e0(j,i) = A_sym_e0(i,j);
	}
      }

      // fill herm matrices
      for (std::size_t i=0; i < myExtent0; ++i) {
	// diagonal has real elements
	A_hem_e0(i,i) = randObj_r();

	for (std::size_t j=i+1; j < myExtent0; ++j) {
	  A_hem_e0(i,j) = {randObj_r(), randObj_i()};
	  A_hem_e0(j,i) = std::conj(A_hem_e0(i,j));
	}
      }

      // fill nonsym matrices and vectors
      fill_random_mdspan(randObj_r, randObj_i, A_e0e1);
      fill_random_mdspan(randObj_r, randObj_i, B_e0e1);
      fill_random_mdspan(randObj_r, randObj_i, x_e0);
      fill_random_mdspan(randObj_r, randObj_i, x_e1);
      fill_random_mdspan(randObj_r, randObj_i, y_e0);
      fill_random_mdspan(randObj_r, randObj_i, z_e0);
    }
    else{
      const auto a = static_cast<value_type>(-5);
      const auto b = static_cast<value_type>( 4);
      UnifDist<value_type> randObj(a, b);

      // fill herm matrices, which for float or double is
      // just a symmetric matrix
      for (std::size_t i=0; i < myExtent0; ++i) {
	for (std::size_t j=i; j < myExtent0; ++j) {
	  A_hem_e0(i,j) = randObj();
	  A_hem_e0(j,i) = A_hem_e0(i,j);
	}
      }

      // fill symmetric matrices
      for (std::size_t i=0; i < myExtent0; ++i) {
	for (std::size_t j=i; j < myExtent0; ++j) {
	  A_sym_e0(i,j) = randObj();
	  A_sym_e0(j,i) = A_sym_e0(i,j);
	}
      }

      // fill nonsym matrices and vectors
      fill_random_mdspan(randObj, A_e0e1);
      fill_random_mdspan(randObj, B_e0e1);
      fill_random_mdspan(randObj, x_e0);
      fill_random_mdspan(randObj, x_e1);
      fill_random_mdspan(randObj, y_e0);
      fill_random_mdspan(randObj, z_e0);
    }
  }

  Kokkos::View<value_type**, Kokkos::HostSpace> A_e0e1_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> B_e0e1_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> A_sym_e0_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> A_hem_e0_view;
  Kokkos::View<value_type*,  Kokkos::HostSpace> x_e0_view;
  Kokkos::View<value_type*,  Kokkos::HostSpace> x_e1_view;
  Kokkos::View<value_type*,  Kokkos::HostSpace> y_e0_view;
  Kokkos::View<value_type*,  Kokkos::HostSpace> z_e0_view;

  using mdspan_r1_t = mdspan<value_type, extents<dynamic_extent>>;
  using mdspan_r2_t = mdspan<value_type, extents<dynamic_extent, dynamic_extent>>;
  mdspan_r2_t A_e0e1; //e0 x e1
  mdspan_r2_t B_e0e1; //e0 x e1
  mdspan_r2_t A_sym_e0; //e0 x e0, symmetric
  mdspan_r2_t A_hem_e0; //e0 x e0, hermitian

  mdspan_r1_t x_e0;  // x vector with extent == e0
  mdspan_r1_t x_e1;  // x vector with extent == e1
  mdspan_r1_t y_e0;  // y vector with extent == e0
  mdspan_r1_t z_e0;  // z vector with extent == e0
};


template<class T>
class _blas3_signed_fixture
  : public _blas2_signed_fixture<T>
{
  using base_t = _blas2_signed_fixture<T>;

  // extents are arbitrarily chosen but not trivially small
  using base_t::myExtent0;
  using base_t::myExtent1;
  const std::size_t myExtent2 = 53;

public:
  using value_type = T;
  using typename base_t::mdspan_r1_t;
  using typename base_t::mdspan_r2_t;

  _blas3_signed_fixture()
    : base_t(),
      B_e0e2_view("B_e0e2_view", myExtent0, myExtent2),
      B_e0e2(B_e0e2_view.data(), myExtent0, myExtent2),
      //
      B_e1e2_view("B_e1e2_view", myExtent1, myExtent2),
      B_e1e2(B_e1e2_view.data(), myExtent1, myExtent2),
      //
      B_e2e1_view("B_e2e1_view", myExtent2, myExtent1),
      B_e2e1(B_e2e1_view.data(), myExtent2, myExtent1),
      //
      C_e0e2_view("C_e0e2_view", myExtent0, myExtent2),
      C_e0e2(C_e0e2_view.data(), myExtent0, myExtent2),
      //
      C_e1e2_view("C_e1e2_view", myExtent1, myExtent2),
      C_e1e2(C_e1e2_view.data(), myExtent1, myExtent2),
      //
      C_e2e0_view("C_e2e0_view", myExtent2, myExtent0),
      C_e2e0(C_e2e0_view.data(), myExtent2, myExtent0),
      //
      E_e0e2_view("E_e0e2_view", myExtent0, myExtent2),
      E_e0e2(E_e0e2_view.data(), myExtent0, myExtent2)
  {

    static_check_value_type(value_type{});

    if constexpr(std::is_same_v<T, std::complex<double>>){
      const auto a_r = static_cast<double>(-5);
      const auto b_r = static_cast<double>( 3);
      UnifDist<double> randObj_r(a_r, b_r);

      const auto a_i = static_cast<double>(-9);
      const auto b_i = static_cast<double>( 7);
      UnifDist<double> randObj_i(a_i, b_i);

      fill_random_mdspan(randObj_r, randObj_i, B_e0e2);
      fill_random_mdspan(randObj_r, randObj_i, B_e1e2);
      fill_random_mdspan(randObj_r, randObj_i, B_e2e1);
      fill_random_mdspan(randObj_r, randObj_i, C_e0e2);
      fill_random_mdspan(randObj_r, randObj_i, C_e1e2);
      fill_random_mdspan(randObj_r, randObj_i, C_e2e0);
      fill_random_mdspan(randObj_r, randObj_i, E_e0e2);
    }
    else{
      const auto a = static_cast<value_type>(-5);
      const auto b = static_cast<value_type>( 4);
      UnifDist<value_type> randObj(a, b);

      fill_random_mdspan(randObj, B_e0e2);
      fill_random_mdspan(randObj, B_e1e2);
      fill_random_mdspan(randObj, B_e2e1);
      fill_random_mdspan(randObj, C_e0e2);
      fill_random_mdspan(randObj, C_e1e2);
      fill_random_mdspan(randObj, C_e2e0);
      fill_random_mdspan(randObj, E_e0e2);
    }
  }

  Kokkos::View<value_type**, Kokkos::HostSpace> B_e0e2_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> B_e1e2_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> B_e2e1_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> C_e0e2_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> C_e1e2_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> C_e2e0_view;
  Kokkos::View<value_type**, Kokkos::HostSpace> E_e0e2_view;

  mdspan_r2_t B_e0e2; //e0 x e2
  mdspan_r2_t B_e1e2; //e1 x e2
  mdspan_r2_t B_e2e1; //e2 x e1

  mdspan_r2_t C_e0e2; //e0 x e2
  mdspan_r2_t C_e1e2; //e1 x e2
  mdspan_r2_t C_e2e0; //e2 x e0

  mdspan_r2_t E_e0e2; //e0 x e2
};

using blas1_signed_float_fixture  = _blas1_signed_fixture<float>;
using blas1_signed_double_fixture = _blas1_signed_fixture<double>;
using blas1_signed_complex_double_fixture = _blas1_signed_fixture<std::complex<double>>;

using blas2_signed_float_fixture  = _blas2_signed_fixture<float>;
using blas2_signed_double_fixture = _blas2_signed_fixture<double>;
using blas2_signed_complex_double_fixture = _blas2_signed_fixture<std::complex<double>>;

using blas3_signed_float_fixture  = _blas3_signed_fixture<float>;
using blas3_signed_double_fixture = _blas3_signed_fixture<double>;
using blas3_signed_complex_double_fixture = _blas3_signed_fixture<std::complex<double>>;

#endif
