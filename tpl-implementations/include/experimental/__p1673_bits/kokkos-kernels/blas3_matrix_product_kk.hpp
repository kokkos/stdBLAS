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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MATRIX_PRODUCT_HPP_

#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"
#include "parallel_matrix.hpp"

namespace KokkosKernelsSTD {

namespace matproduct_impl {

void runtime_error(const std::string_view func_name, const std::string_view msg)
{
  std::string message = std::string("KokkosBlas: ")
      + std::string(func_name) + std::string(": ")
      + std::string(msg);
  throw std::runtime_error(message.c_str());
}

template <class AType, class Triangle, class BType, class CType>
void check_product(AType A, Triangle t, BType B, CType C,
                   const std::string_view func_name)
{
  // P1673 constraints (redundant to mdspan extents function headers)
  static_assert(A.rank() == 2);
  static_assert(B.rank() == 2);
  static_assert(C.rank() == 2);
  using a_layout = typename AType::layout_type;
  static_assert(Impl::triangle_layout_match_v<a_layout, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));

  // P1673 preconditions
  if (A.extent(0) != A.extent(1)) {
    runtime_error(func_name, "A.extent(0) != A.extent(1)");
  }
}

template <class EType, class CType>
void check_E(EType E, CType C, const std::string_view func_name)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(E.rank() == 2);

  // P1673 mandates
  static_assert(Impl::static_extent_match(E.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(E.static_extent(1), C.static_extent(1)));

  // P1673 preconditions
  if ( E.extent(0) != C.extent(0) ){
    runtime_error(func_name, "E.extent(0) != C.extent(0)");
  }
  if ( E.extent(1) != C.extent(1) ){
    runtime_error(func_name, "E.extent(1) != C.extent(1)");
  }
}

template <class AType, class Triangle, class BType, class CType>
void check_left_product(AType A, Triangle t, BType B, CType C,
                        const std::string_view func_name)
{
  check_product(A, t, B, C, func_name);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(1), B.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(B.static_extent(1), C.static_extent(1)));

  // P1673 preconditions
  if (A.extent(1) != B.extent(0)) {
    runtime_error(func_name, "A.extent(1) != B.extent(0)");
  }
  if (A.extent(0) != C.extent(0)) {
    runtime_error(func_name, "A.extent(0) != C.extent(0)");
  }
  if (B.extent(1) != C.extent(1)) {
    runtime_error(func_name, "B.extent(1) != C.extent(1)");
  }
}

template <class AType, class Triangle, class BType, class EType, class CType>
void check_left_product(AType A, Triangle t, BType B, EType E, CType C,
                        const std::string_view func_name)
{
  check_left_product(A, t, B, C, func_name);
  check_E(E, C, func_name);
}

template <class AType, class Triangle, class BType, class CType>
void check_right_product(AType A, Triangle t, BType B, CType C,
                        const std::string_view func_name)
{
  check_product(A, t, B, C, func_name);

  // P1673 mandates
  static_assert(Impl::static_extent_match(B.static_extent(1), A.static_extent(0)));
  static_assert(Impl::static_extent_match(B.static_extent(0), C.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(1), C.static_extent(1)));

  // P1673 preconditions
  if (B.extent(1) != A.extent(0)) {
    runtime_error(func_name, "B.extent(1) != A.extent(0)");
  }
  if (B.extent(0) != C.extent(0)) {
    runtime_error(func_name, "B.extent(0) != C.extent(0)");
  }
  if (A.extent(1) != C.extent(1)) {
    runtime_error(func_name, "A.extent(1) != C.extent(1)");
  }
}

template <class AType, class Triangle, class BType, class EType, class CType>
void check_right_product(AType A, Triangle t, BType B, EType E, CType C,
                        const std::string_view func_name)
{
  check_right_product(A, t, B, C, func_name);
  check_E(E, C, func_name);
}

template <class ExecSpace,
          class AType,
          class BType,
          class CType,
          class InitFunc,
          class UpdateFunc>
void product(ExecSpace &&exec, AType A, BType B, CType C, InitFunc init, UpdateFunc update)
{
  const auto A_view = Impl::mdspan_to_view(A);
  const auto B_view = Impl::mdspan_to_view(B);
  auto C_view = Impl::mdspan_to_view(C);

  using c_element_type = typename decltype(C_view)::non_const_value_type;
  using size_type = typename std::experimental::extents<>::size_type;

  KokkosKernelsSTD::Impl::ParallelMatrixVisitor(
        std::move(exec), C_view).for_each_matrix_element(
    KOKKOS_LAMBDA(const auto i, const auto j) {
      decltype(auto) cij = C_view(i, j);
      init(cij, i, j);
      const size_type ext1 = A_view.extent(1);
      for (size_type k = 0; k < ext1; ++k) {
        update(i, j, k, cij, A_view, B_view);
      }
    });
}

template <class ExecSpace,
          class AType,
          class BType,
          class CType,
          class UpdateFunc>
void product(ExecSpace &&exec, AType A, BType B, CType C, UpdateFunc update)
{
  product(std::move(exec), A, B, C,
    KOKKOS_LAMBDA(auto &&cij, const auto i, const auto j) {
      using c_element_type = std::remove_cvref_t<decltype(cij)>;
      cij = c_element_type{}; // zero
    }, update);
}

template <class ExecSpace,
          class AType,
          class BType,
          class EType,
          class ElementType_C,
          std::experimental::extents<>::size_type numRows_C,
          std::experimental::extents<>::size_type numCols_C,
          class Layout_C,
          class UpdateFunc>
void product(ExecSpace &&exec, AType A, BType B, EType E,
             std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
                std::experimental::default_accessor<ElementType_C>> C, // disambiguate with init()
             UpdateFunc update)
{
  const auto E_view = Impl::mdspan_to_view(E);
  product(std::move(exec), A, B, C,
    KOKKOS_LAMBDA(auto &&cij, const auto i, const auto j) {
      cij = E_view(i, j);
    }, update);
}

} // namespace matproduct_impl

// Overwriting symmetric matrix-matrix left product
// performs BLAS xSYMM: C = A x B

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    /* requires */ (Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void symmetric_matrix_left_product(
  kokkos_exec<ExecSpace>&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C)
{
  matproduct_impl::check_left_product(A, t, B, C, "symmetric_matrix_left_product");
 
  Impl::signal_kokkos_impl_called("overwriting_symmetric_matrix_left_product");

  constexpr bool lower = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;

  matproduct_impl::product(ExecSpace(), A, B, C,
    KOKKOS_LAMBDA(const auto i, const auto j, const auto k,
                  auto &&cij, auto A_view, auto B_view) {
      const bool flip = lower ? i <= k : i >= k;
      const auto aik = flip ? A_view(k, i) : A_view(i, k);
      cij += aik * B_view(k, j);
    });
}

// Updating symmetric matrix-matrix left product
// performs BLAS xSYMM: C = E + A x B

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_E,
    std::experimental::extents<>::size_type numRows_E,
    std::experimental::extents<>::size_type numCols_E,
    class Layout_E,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    /* requires */ (Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
        and Impl::is_unique_layout_v<Layout_E, numRows_E, numCols_E>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void symmetric_matrix_left_product(
  kokkos_exec<ExecSpace>&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_E, std::experimental::extents<numRows_E, numCols_E>, Layout_E,
    std::experimental::default_accessor<ElementType_E>> E,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C)
{
  matproduct_impl::check_left_product(A, t, B, E, C, "symmetric_matrix_left_product");

  Impl::signal_kokkos_impl_called("updating_symmetric_matrix_left_product");

  constexpr bool lower = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;

  matproduct_impl::product(ExecSpace(), A, B, E, C,
    KOKKOS_LAMBDA(const auto i, const auto j, const auto k,
        auto &&cij, auto A_view, auto B_view) {
      const bool flip = lower ? i <= k : i >= k;
      const auto aik = flip ? A_view(k, i) : A_view(i, k);
      cij += aik * B_view(k, j);
    });
}

// Overwriting symmetric matrix-matrix right product
// performs BLAS xSYMM: C = B x A

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    /* requires */ (Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void symmetric_matrix_right_product(
  kokkos_exec<ExecSpace>&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C)
{
  matproduct_impl::check_right_product(A, t, B, C, "symmetric_matrix_right_product");
 
  Impl::signal_kokkos_impl_called("overwriting_symmetric_matrix_right_product");

  constexpr bool lower = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;

  matproduct_impl::product(ExecSpace(), A, B, C,
    KOKKOS_LAMBDA(const auto i, const auto j, const auto k,
                  auto &&cij, auto A_view, auto B_view) {
      const bool flip = lower ? j >= k : j <= k;
      const auto akj = flip ? A_view(j, k) : A_view(k, j);
      cij += B(i, k) * akj;
    });
}

// Updating symmetric matrix-matrix right product
// performs BLAS xSYMM: C = E + B x A

MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_E,
    std::experimental::extents<>::size_type numRows_E,
    std::experimental::extents<>::size_type numCols_E,
    class Layout_E,
    class ElementType_C,
    std::experimental::extents<>::size_type numRows_C,
    std::experimental::extents<>::size_type numCols_C,
    class Layout_C,
    /* requires */ (Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and Impl::is_unique_layout_v<Layout_C, numRows_C, numCols_C>
        and Impl::is_unique_layout_v<Layout_E, numRows_E, numCols_E>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void symmetric_matrix_right_product(
  kokkos_exec<ExecSpace>&& /* exec */,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B,
    std::experimental::default_accessor<ElementType_B>> B,
  std::experimental::mdspan<ElementType_E, std::experimental::extents<numRows_E, numCols_E>, Layout_E,
    std::experimental::default_accessor<ElementType_E>> E,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C,
    std::experimental::default_accessor<ElementType_C>> C)
{
  matproduct_impl::check_right_product(A, t, B, E, C, "symmetric_matrix_right_product");

  Impl::signal_kokkos_impl_called("updating_symmetric_matrix_right_product");

  constexpr bool lower = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;

  matproduct_impl::product(ExecSpace(), A, B, E, C,
    KOKKOS_LAMBDA(const auto i, const auto j, const auto k,
                  auto &&cij, auto A_view, auto B_view) {
      const bool flip = lower ? j >= k : j <= k;
      const auto akj = flip ? A_view(j, k) : A_view(k, j);
      cij += B(i, k) * akj;
    });
}

} // namespace KokkosKernelsSTD
#endif
