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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_

#include <complex>
#include <type_traits>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

// Overwriting general matrix-vector product: y := A * x
namespace {
template <typename Exec, typename A_t, typename x_t, typename y_t, typename = void>
struct is_custom_matrix_vector_product_avail : std::false_type {};

template <typename Exec, typename A_t, typename x_t, typename y_t>
struct is_custom_matrix_vector_product_avail<Exec, A_t, x_t, y_t,
                   std::void_t<decltype(matrix_vector_product(
                              std::declval<Exec>(),
                              std::declval<A_t>(),
                              std::declval<x_t>(),
                              std::declval<y_t>())) >> {
                   static constexpr bool value = !std::is_same<Exec,std::experimental::linalg::impl::inline_exec_t>::value;
};


template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
  requires (Layout_A::template mapping<extents<numRows_A, numCols_A> >::is_always_unique())
void linalg_matrix_vector_product(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
    for (size_type j = 0; j < A.extent(1); ++j) {
      y(i) += A(i,j) * x(j);
    }
  }
}

} // end anonymous namespace

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
void matrix_vector_product(
  ExecutionPolicy&& exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  if constexpr(is_custom_matrix_vector_product_avail<
       decltype(execpolicy_mapper(std::declval<ExecutionPolicy>())),
       decltype(A),
       decltype(x),
       decltype(y)>::value) {
    matrix_vector_product(execpolicy_mapper(exec), A, x, y);
  } else {
    linalg_matrix_vector_product(std::experimental::linalg::impl::inline_exec_t(), A, x, y);
  }
}

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
void matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  matrix_vector_product(std::experimental::linalg::impl::default_exec_t(), A, x, y);
}

// Updating general matrix-vector product: z := y + A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = z(i);
    for (size_type j = 0; j < A.extent(1); ++j) {
      y(i) += A(i,j) * x(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
void matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  matrix_vector_product(A, x, y, z);
}

// Overwriting symmetric matrix-vector product: y := A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void symmetric_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += A_ij * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += A_ij * x(i);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
void symmetric_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  symmetric_matrix_vector_product(A, t, x, y);
}

// Updating symmetric matrix-vector product: z := y + A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void symmetric_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += A_ij * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += A_ij * x(i);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
void symmetric_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  symmetric_matrix_vector_product(A, t, x, y, z);
}

// Overwriting Hermitian matrix-vector product: y := A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void hermitian_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }

  using std::conj;
  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += conj(A_ij) * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += conj(A_ij) * x(i);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void hermitian_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  hermitian_matrix_vector_product(A, t, x, y);
}

// Updating Hermitian matrix-vector product: z := y + A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void hermitian_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }

  using std::conj;
  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += conj(A_ij) * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += conj(A_ij) * x(i);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
void hermitian_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  hermitian_matrix_vector_product(A, t, x, y, z);
}

// Overwriting triangular matrix-vector product: y := A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void triangular_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }
  constexpr bool explicitDiagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_lower = explicitDiagonal ? j : j + size_type(1);
      for (size_type i = i_lower; i < A.extent(0); ++i) {
        y(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        y(j) += /* 1 times */ x(j);
      }
    }
  }
  else {
    for (ptrdiff_t j = 0; j < A.extent(1); ++j) {
      const ptrdiff_t i_upper = explicitDiagonal ? j : j - 1;
      for (ptrdiff_t i = 0; i <= i_upper; ++i) {
        y(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        y(j) += /* 1 times */ x(j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y>
void triangular_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y)
{
  triangular_matrix_vector_product(A, t, d, x, y);
}

// Updating triangular matrix-vector product: z := y + A * x

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
  requires (Layout_A::template mapping<std::experimental::extents<numRows_A, numCols_A>>::is_always_unique())
void triangular_matrix_vector_product(
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  using size_type = typename extents<>::size_type;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }
  constexpr bool explicitDiagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_lower = explicitDiagonal ? j : j + size_type(1);
      for (size_type i = i_lower; i < A.extent(0); ++i) {
        z(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        z(j) += /* 1 times */ x(j);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const ptrdiff_t i_upper = explicitDiagonal ? j : j - size_type(1);
      for (size_type i = 0; i <= i_upper; ++i) {
        z(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        z(j) += /* 1 times */ x(j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         extents<>::size_type ext_z,
         class Layout_z,
         class Accessor_z>
void triangular_matrix_vector_product(
  ExecutionPolicy&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  triangular_matrix_vector_product(A, t, d, x, y, z);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_
