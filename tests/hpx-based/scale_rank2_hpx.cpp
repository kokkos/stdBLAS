//  Copyright (c) 2022 Hartmut Kaiser

#include <complex>
#include <experimental/linalg>
#include <experimental/mdspan>

#include "gtest/gtest.h"
#include "gtest_fixtures.hpp"

namespace {

template <class A_t, class FactorT>
void scale_gold_solution(A_t A, FactorT factor)
{
    FactorT result = {};
    for (std::size_t i = 0; i < A.extent(0); ++i)
    {
        for (std::size_t j = 0; j < A.extent(1); ++j)
        {
            A(i, j) *= factor;
        }
    }
}

template <class ExPolicy, class A_t, class FactorT>
void hpx_blas_scale_test_impl(ExPolicy policy, A_t A, FactorT factor)
{
    namespace stdla = std::experimental::linalg;

    using value_type = typename A_t::value_type;
    const std::size_t extent0 = A.extent(0);
    const std::size_t extent1 = A.extent(1);

    // compute gold
    std::vector<value_type> gold(extent0 * extent1);
    using mdspan_t =
        mdspan<value_type, extents<dynamic_extent, dynamic_extent>>;
    mdspan_t A_gold(gold.data(), extent0, extent1);
    for (std::size_t i = 0; i < extent0; ++i)
    {
        for (std::size_t j = 0; j < extent1; ++j)
        {
            A_gold(i, j) = A(i, j);
        }
    }
    scale_gold_solution(A_gold, factor);

    stdla::scale(policy, factor, A);

    if constexpr (std::is_same_v<value_type, float>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_FLOAT_EQ(A(i, j), A_gold(i, j));
            }
        }
    }

    if constexpr (std::is_same_v<value_type, double>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_DOUBLE_EQ(A(i, j), A_gold(i, j));
            }
        }
    }

    if constexpr (std::is_same_v<value_type, std::complex<double>>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_DOUBLE_EQ(A(i, j).real(), A_gold(i, j).real());
                EXPECT_DOUBLE_EQ(A(i, j).imag(), A_gold(i, j).imag());
            }
        }
    }
}
}    // namespace

TEST_F(blas2_signed_float_fixture, hpx_scale)
{
    hpx_blas_scale_test_impl(
        HPXKernelsSTD::hpx_exec<>(), A_e0e1, static_cast<value_type>(2));
    hpx_blas_scale_test_impl(
        hpx::execution::par, A_e0e1, static_cast<value_type>(2));
    hpx_blas_scale_test_impl(
        hpx::execution::par_unseq, A_e0e1, static_cast<value_type>(2));
    //#if defined(HPX_HAVE_DATAPAR)
    //  hpx_blas_scale_test_impl(hpx::execution::simd, A_e0e1, static_cast<value_type>(2));
    //  hpx_blas_scale_test_impl(hpx::execution::par_simd, A_e0e1, static_cast<value_type>(2));
    //#endif
}

TEST_F(blas2_signed_double_fixture, hpx_scale)
{
    hpx_blas_scale_test_impl(
        HPXKernelsSTD::hpx_exec<>(), A_e0e1, static_cast<value_type>(2));
    hpx_blas_scale_test_impl(
        hpx::execution::par, A_e0e1, static_cast<value_type>(2));
    hpx_blas_scale_test_impl(
        hpx::execution::par_unseq, A_e0e1, static_cast<value_type>(2));
    //#if defined(HPX_HAVE_DATAPAR)
    //  hpx_blas_scale_test_impl(hpx::execution::simd, A_e0e1, static_cast<value_type>(2));
    //  hpx_blas_scale_test_impl(hpx::execution::par_simd, A_e0e1, static_cast<value_type>(2));
    //#endif
}

TEST_F(blas2_signed_complex_double_fixture, hpx_scale_complex_factor)
{
    using kc_t = std::complex<double>;
    using stdc_t = value_type;
    if constexpr (alignof(value_type) == alignof(kc_t))
    {
        const value_type factor{2., 0.};
        hpx_blas_scale_test_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, factor);
        hpx_blas_scale_test_impl(hpx::execution::par, A_e0e1, factor);
        hpx_blas_scale_test_impl(hpx::execution::par_unseq, A_e0e1, factor);
        //#if defined(HPX_HAVE_DATAPAR)
        //    hpx_blas_scale_test_impl(hpx::execution::simd, A_e0e1, factor);
        //    hpx_blas_scale_test_impl(hpx::execution::par_simd, A_e0e1, factor);
        //#endif
    }
}

TEST_F(blas2_signed_complex_double_fixture, hpx_scale_double_factor)
{
    using kc_t = std::complex<double>;
    using stdc_t = value_type;
    if constexpr (alignof(value_type) == alignof(kc_t))
    {
        hpx_blas_scale_test_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, 2.);
        hpx_blas_scale_test_impl(hpx::execution::par, A_e0e1, 2.);
        hpx_blas_scale_test_impl(hpx::execution::par_unseq, A_e0e1, 2.);
        //#if defined(HPX_HAVE_DATAPAR)
        //    hpx_blas_scale_test_impl(hpx::execution::simd, A_e0e1, 2.);
        //    hpx_blas_scale_test_impl(hpx::execution::par_simd, A_e0e1, 2.);
        //#endif
    }
}
