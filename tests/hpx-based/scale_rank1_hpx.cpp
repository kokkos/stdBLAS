//  Copyright (c) 2022 Hartmut Kaiser

#include <complex>
#include <experimental/linalg>
#include <experimental/mdspan>

#include "gtest/gtest.h"
#include "gtest_fixtures.hpp"

namespace {

template <class x_t, class FactorT>
void scale_gold_solution(x_t x, FactorT factor)
{
    FactorT result = {};
    for (std::size_t i = 0; i < x.extent(0); ++i)
    {
        x(i) *= factor;
    }
}

template <class ExPolicy, class x_t, class FactorT>
void hpx_blas1_scale_test_impl(ExPolicy policy, x_t x, FactorT factor)
{
    namespace stdla = std::experimental::linalg;

    using value_type = typename x_t::value_type;
    const std::size_t extent = x.extent(0);

    // compute gold
    std::vector<value_type> gold(extent);
    using mdspan_t = mdspan<value_type, extents<dynamic_extent>>;
    mdspan_t x_gold(gold.data(), extent);
    for (std::size_t i = 0; i < x.extent(0); ++i)
    {
        x_gold(i) = x(i);
    }
    scale_gold_solution(x_gold, factor);

    stdla::scale(policy, factor, x);

    if constexpr (std::is_same_v<value_type, float>)
    {
        for (std::size_t i = 0; i < extent; ++i)
        {
            EXPECT_FLOAT_EQ(x(i), x_gold(i));
        }
    }

    if constexpr (std::is_same_v<value_type, double>)
    {
        for (std::size_t i = 0; i < extent; ++i)
        {
            EXPECT_DOUBLE_EQ(x(i), x_gold(i));
        }
    }

    if constexpr (std::is_same_v<value_type, std::complex<double>>)
    {
        for (std::size_t i = 0; i < extent; ++i)
        {
            EXPECT_DOUBLE_EQ(x(i).real(), x_gold(i).real());
            EXPECT_DOUBLE_EQ(x(i).imag(), x_gold(i).imag());
        }
    }
}
}    // namespace

TEST_F(blas1_signed_float_fixture, hpx_scale)
{
    hpx_blas1_scale_test_impl(
        HPXKernelsSTD::hpx_exec<>(), x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par, x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par_unseq, x, static_cast<value_type>(2));
#if defined(HPX_HAVE_DATAPAR)
    hpx_blas1_scale_test_impl(
        hpx::execution::simd, x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par_simd, x, static_cast<value_type>(2));
#endif
}

TEST_F(blas1_signed_double_fixture, hpx_scale)
{
    hpx_blas1_scale_test_impl(
        HPXKernelsSTD::hpx_exec<>(), x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par, x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par_unseq, x, static_cast<value_type>(2));
#if defined(HPX_HAVE_DATAPAR)
    hpx_blas1_scale_test_impl(
        hpx::execution::simd, x, static_cast<value_type>(2));
    hpx_blas1_scale_test_impl(
        hpx::execution::par_simd, x, static_cast<value_type>(2));
#endif
}

TEST_F(blas1_signed_complex_double_fixture, hpx_scale_complex_factor)
{
    using kc_t = std::complex<double>;
    using stdc_t = value_type;
    if constexpr (alignof(value_type) == alignof(kc_t))
    {
        const value_type factor{2., 0.};
        hpx_blas1_scale_test_impl(HPXKernelsSTD::hpx_exec<>(), x, factor);
        hpx_blas1_scale_test_impl(hpx::execution::par, x, factor);
        hpx_blas1_scale_test_impl(hpx::execution::par_unseq, x, factor);
#if defined(HPX_HAVE_DATAPAR)
        hpx_blas1_scale_test_impl(hpx::execution::simd, x, factor);
        hpx_blas1_scale_test_impl(hpx::execution::par_simd, x, factor);
#endif
    }
}

TEST_F(blas1_signed_complex_double_fixture, hpx_scale_double_factor)
{
    using kc_t = std::complex<double>;
    using stdc_t = value_type;
    if constexpr (alignof(value_type) == alignof(kc_t))
    {
        hpx_blas1_scale_test_impl(HPXKernelsSTD::hpx_exec<>(), x, 2.);
        hpx_blas1_scale_test_impl(hpx::execution::par, x, 2.);
        hpx_blas1_scale_test_impl(hpx::execution::par_unseq, x, 2.);
#if defined(HPX_HAVE_DATAPAR)
        hpx_blas1_scale_test_impl(hpx::execution::simd, x, 2.);
        hpx_blas1_scale_test_impl(hpx::execution::par_simd, x, 2.);
#endif
    }
}
