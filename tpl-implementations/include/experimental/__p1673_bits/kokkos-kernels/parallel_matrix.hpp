 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_PARALLEL_MATRIX_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_PARALLEL_MATRIX_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

// manages parallel execution of independent action
// called like action(i, j) for each matrix element A(i, j)
template <typename ExecSpace, typename MatrixType>
class ParallelMatrixVisitor {
public:
  KOKKOS_INLINE_FUNCTION ParallelMatrixVisitor(ExecSpace &&exec_in, MatrixType A_in):
    exec(exec_in), A(A_in), ext0(A.extent(0)), ext1(A.extent(1))
  {}

  template <typename ActionType>
  KOKKOS_INLINE_FUNCTION
  void for_each_matrix_element(ActionType action) {
    if (ext0 > ext1) { // parallel rows
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext0),
        KOKKOS_LAMBDA(const auto i) {
          using idx_type = std::remove_const_t<decltype(i)>;
          for (idx_type j = 0; j < ext1; ++j) {
            action(i, j);
          }
        });
    } else { // parallel columns
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext1),
        KOKKOS_LAMBDA(const auto j) {
          using idx_type = std::remove_const_t<decltype(j)>;
          for (idx_type i = 0; i < ext0; ++i) {
            action(i, j);
          }
        });
    }
    exec.fence();
  }

  template <typename ActionType>
  void for_each_triangle_matrix_element(std::experimental::linalg::upper_triangle_t t, ActionType action) {
    Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext1),
      KOKKOS_LAMBDA(const auto j) {
        using idx_type = std::remove_const_t<decltype(j)>;
        for (idx_type i = 0; i <= j; ++i) {
          action(i, j);
        }
      });
    exec.fence();
  }

  template <typename ActionType>
  void for_each_triangle_matrix_element(std::experimental::linalg::lower_triangle_t t, ActionType action) {
    for_each_triangle_matrix_element(std::experimental::linalg::upper_triangle,
        [action](const auto i, const auto j) {
          action(j, i);
      });
  }

private:
  ExecSpace exec;
  MatrixType A;
  size_t ext0;
  size_t ext1;
};

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
