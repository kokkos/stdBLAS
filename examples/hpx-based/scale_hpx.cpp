//  Copyright (c) 2022 Hartmut Kaiser

#include <hpx/execution.hpp>
#include <hpx/hpx_main.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include <cstddef>
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "dot example: calling hpx-kernels" << std::endl;

    std::size_t N = 40;
    {
        std::vector<double> data(N);
        double* a_ptr = data.data();

        // Requires CTAD working, GCC 11.1 works but some others are buggy
        // std::experimental::mdspan a(a_ptr,N);
        std::experimental::mdspan<double,
            std::experimental::extents<std::size_t, std::experimental::dynamic_extent>>
            a(a_ptr, N);
        for (std::size_t i = 0; i < a.extent(0); i++)
            a(i) = double(i);

        // This forwards to HPXKernels
        std::experimental::linalg::scale(HPXKernelsSTD::hpx_exec<>(), 2.0, a);
        // This forwards to HPXKernels if LINALG_ENABLE_HPX_DEFAULT is ON
        std::experimental::linalg::scale(std::execution::par, 2.0, a);
        // This always forwards to HPXKernels
        std::experimental::linalg::scale(hpx::execution::par, 2.0, a);
        // This goes to the base implementation
        std::experimental::linalg::scale(std::execution::seq, 2.0, a);
        // This also goes to the base implementation
        std::experimental::linalg::scale(hpx::execution::seq, 2.0, a);
#if defined(HPX_HAVE_DATAPAR)
        // this invokes a explicitly vectorized version
        std::experimental::linalg::scale(hpx::execution::simd, 2.0, a);
        std::experimental::linalg::scale(hpx::execution::par_simd, 2.0, a);
#endif
        for (std::size_t i = 0; i < a.extent(0); i++)
            printf("%zi %lf\n", i, a(i));
    }
    return 0;
}
