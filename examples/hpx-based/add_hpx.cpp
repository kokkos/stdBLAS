//  Copyright (c) 2022 Hartmut Kaiser

#include <hpx/execution.hpp>
#include <hpx/hpx_main.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include <cstddef>
#include <execution>
#include <iostream>
#include <vector>

template <class T1, class ScalarType>
void print_elements(
    const T1& v, const std::vector<ScalarType>& gold, char const* policy_str)
{
    std::cout << "Using policy: " << policy_str << "\n";
    for (std::size_t i = 0; i < v.size(); i++)
    {
        std::cout << "computed = " << v(i) << " , gold = " << gold[i] << "\n";
    }
}

void reset(auto z)
{
    for (std::size_t i = 0; i < z.extent(0); i++)
    {
        z(i) = 0;
    }
}

int main(int argc, char* argv[])
{
    std::cout << "add example: calling HPX-kernels" << std::endl;

    std::size_t N = 50;

    using value_type = double;

    std::vector<value_type> x_data(N);
    std::vector<value_type> y_data(N);
    std::vector<value_type> z_data(N);

    value_type* x_ptr = x_data.data();
    value_type* y_ptr = y_data.data();
    value_type* z_ptr = z_data.data();

    using dyn_1d_ext_type = std::experimental::extents<std::size_t,
        std::experimental::dynamic_extent>;
    using mdspan_type = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type x(x_ptr, N);
    mdspan_type y(y_ptr, N);
    mdspan_type z(z_ptr, N);

    std::vector<value_type> gold(N);
    for (std::size_t i = 0; i < x.extent(0); i++)
    {
        x(i) = static_cast<value_type>(i);
        y(i) = i + static_cast<value_type>(10);
        z(i) = 0;
        gold[i] = x(i) + y(i);
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value = 2.0;

    // This goes to the base implementation
    {
        stdla::add(std::execution::seq, x, y, z);
        print_elements(z, gold, "std::execution::seq");
    }

    // This also goes to the base implementation
    {
        reset(z);    // reset z since it is modified above
        stdla::add(hpx::execution::seq, x, y, z);
        print_elements(z, gold, "hpx::execution::seq");
    }

    // This forwards to HPXKernels
    {
        reset(z);    // reset z since it is modified above
        stdla::add(HPXKernelsSTD::hpx_exec<>(), x, y, z);
        print_elements(z, gold, "HPXKernelsSTD::hpx_exec<>()");
    }

    // This forwards to HPXKernels if LINALG_ENABLE_HPX_DEFAULT is ON
    {
        reset(z);    // reset z since it is modified above
        stdla::add(std::execution::par, x, y, z);
        print_elements(z, gold, "std::execution::par");
    }

    // This forwards to HPXKernels
    {
        reset(z);    // reset z since it is modified above
        stdla::add(hpx::execution::par, x, y, z);
        print_elements(z, gold, "hpx::execution::par");
    }

#if defined(HPX_HAVE_DATAPAR)
    // this invokes a explicitly vectorized HPX versions
    {
        reset(z);    // reset z since it is modified above
        stdla::add(hpx::execution::simd, x, y, z);
        print_elements(z, gold, "hpx::execution::simd");
    }

    {
        reset(z);    // reset z since it is modified above
        stdla::add(hpx::execution::par_simd, x, y, z);
        print_elements(z, gold, "hpx::execution::par_simd");
    }
#endif

    return 0;
}
