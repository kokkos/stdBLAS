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

  std::size_t N = 40;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> a_view("A",N);
    double* a_ptr = a_view.data();

    // Requires CTAD working, GCC 11.1 works but some others are buggy
    // std::experimental::mdspan a(a_ptr,N);
    std::experimental::mdspan<double,std::experimental::extents<std::experimental::dynamic_extent>> a(a_ptr,N);
    for(std::size_t i=0; i<a.extent(0); i++) a(i) = i;

    // This forwards to KokkosKernels (https://github.com/kokkos/kokkos-kernels
    std::experimental::linalg::scale(KokkosKernelsSTD::kokkos_exec<>(),2.0,a);
    // This forwards to KokkosKernels if LINALG_ENABLE_KOKKOS_DEFAULT is ON
    std::experimental::linalg::scale(std::execution::par,2.0,a);
    // This goes to the base implementation
    std::experimental::linalg::scale(std::execution::seq,2.0,a);
    for(std::size_t i=0; i<a.extent(0); i++) printf("%i %lf\n",i,a(i));
  }
  Kokkos::finalize();
}
