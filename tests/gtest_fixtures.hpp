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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_

#include <experimental/mdspan>

#include "gtest/gtest.h"
#include <vector>

  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using dbl_vector_t = mdspan<double, extents<dynamic_extent>>;
  using cpx_vector_t = mdspan<std::complex<double>, extents<dynamic_extent>>;
  constexpr ptrdiff_t NROWS(10);

  // 1-norm:   4.6
  // inf-norm: 0.9
  class unsigned_double_vector : public ::testing::Test {
    protected:
      unsigned_double_vector() :
        storage(10),
        v(storage.data(), 10)
      {
        v(0) = 0.5;  
        v(1) = 0.2;  
        v(2) = 0.1;
        v(3) = 0.4;
        v(4) = 0.8;
        v(5) = 0.7;
        v(6) = 0.3;
        v(7) = 0.5;
        v(8) = 0.2;
        v(9) = 0.9;
      }
    
      std::vector<double> storage;
      dbl_vector_t v;
  }; // end class unsigned_double_vector

  // 1-norm:   4.6
  // inf-norm: 0.9
  class signed_double_vector : public ::testing::Test {
    protected:
      signed_double_vector() :
        storage(10),
        v(storage.data(), 10)
      {
        v(0) =  0.5;  
        v(1) =  0.2;  
        v(2) =  0.1;
        v(3) =  0.4;
        v(4) = -0.8;
        v(5) = -0.7;
        v(6) = -0.3;
        v(7) =  0.5;
        v(8) =  0.2;
        v(9) = -0.9;
      }
    
      std::vector<double> storage;
      dbl_vector_t v;
  }; // end class signed_double_vector

  // 1-norm:   3.5188912597625004
  // 2-norm:   1.6673332000533068
  // inf-norm: 1.063014581273465
  class signed_complex_vector : public ::testing::Test {
    protected:
      signed_complex_vector() :
        storage(5),
        v(storage.data(), 5)
      {
        v(0) = std::complex<double>( 0.5,  0.2);
        v(1) = std::complex<double>( 0.1,  0.4);
        v(2) = std::complex<double>(-0.8, -0.7);
        v(3) = std::complex<double>(-0.3,  0.5);
        v(4) = std::complex<double>( 0.2, -0.9);
      }
    
      std::vector<std::complex<double>> storage;
      cpx_vector_t v;
  }; // end class signed_double_vector

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_
