//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

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
