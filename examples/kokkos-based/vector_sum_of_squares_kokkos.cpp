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
  std::cout << "vector_sum_of_squares example: calling kokkos-kernels" << std::endl;

  std::size_t N = 20;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;
    Kokkos::View<value_type*> x_view("x",N);
    value_type* x_ptr = x_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::experimental::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type x(x_ptr,N);
    for(std::size_t i=0; i<x.extent(0); i++){
      x(i) = i;
    }

    namespace stdla = std::experimental::linalg;
    stdla::sum_of_squares_result<value_type> init_value{1., 1.};

    const auto res = stdla::vector_sum_of_squares(x, init_value);
    std::cout << "Default result: " << res.scaling_factor << " " << res.scaled_sum_of_squares << '\n';

    // FRIZZI: Oct 27: kk currently not impl yet, just placeholder to ensure hook forwards correctly
    const auto res_kk = stdla::vector_sum_of_squares(KokkosKernelsSTD::kokkos_exec<>(), x, init_value);
    (void)res_kk;
    //std::cout << "Kokkos result: " << res_kk.scaling_factor << " " << res_kk.scaled_sum_of_squares << '\n';

  }
  Kokkos::finalize();
}
