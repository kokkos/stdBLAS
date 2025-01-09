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

#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "dot example: calling kokkos-kernels" << std::endl;

  std::size_t N = 50;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;

    Kokkos::View<value_type*> a_view("A",N);
    Kokkos::View<value_type*> b_view("B",N);
    value_type* a_ptr = a_view.data();
    value_type* b_ptr = b_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::experimental::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type a(a_ptr,N);
    mdspan_type b(b_ptr,N);
    for(std::size_t i=0; i<a.extent(0); i++){
      a(i) = i;
      b(i) = i;
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value(2.0);

    // This goes to the base implementation
    const auto res_seq = stdla::dot(std::execution::seq, a, b, init_value);
    printf("Seq result    = %lf\n", res_seq);

    // This forwards to KokkosKernels
    const auto res_kk = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(), a, b, init_value);
    printf("Kokkos result = %lf\n", res_kk);
  }
  Kokkos::finalize();
}
