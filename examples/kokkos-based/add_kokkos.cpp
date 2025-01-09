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

template<class T1, class ScalarType>
void print_elements(const T1 & v, const std::vector<ScalarType> & gold)
{
  for(std::size_t i=0; i<v.size(); i++){
    std::cout << "computed = " << v(i)
	      << " , gold = "
	      << gold[i]
	      << "\n";
  }
}

int main(int argc, char* argv[])
{
  std::cout << "add example: calling kokkos-kernels" << std::endl;

  std::size_t N = 50;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;

    Kokkos::View<value_type*> x_view("x",N);
    Kokkos::View<value_type*> y_view("y",N);
    Kokkos::View<value_type*> z_view("z",N);

    value_type* x_ptr = x_view.data();
    value_type* y_ptr = y_view.data();
    value_type* z_ptr = z_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::experimental::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type x(x_ptr,N);
    mdspan_type y(y_ptr,N);
    mdspan_type z(z_ptr,N);

    std::vector<value_type> gold(N);
    for(std::size_t i=0; i<x.extent(0); i++){
      x(i) = i;
      y(i) = i + static_cast<value_type>(10);
      z(i) = 0;
      gold[i] = x(i) + y(i);
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value = 2.0;

    {
      // This goes to the base implementation
      stdla::add(std::execution::seq, x, y, z);
    }

    {
      // reset z since it is modified above
      for(std::size_t i=0; i<z.extent(0); i++){ z(i) = 0; }

      // This forwards to KokkosKernels
      stdla::add(KokkosKernelsSTD::kokkos_exec<>(), x,y,z);
      print_elements(z, gold);
    }

  }
  Kokkos::finalize();
}
