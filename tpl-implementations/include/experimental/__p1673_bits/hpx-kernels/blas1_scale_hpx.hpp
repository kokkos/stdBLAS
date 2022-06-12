//  Copyright (c) 2022 Hartmut Kaiser

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_SCALE_HPP_

#include <hpx/algorithm.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include "signal_hpx_impl_called.hpp"

namespace HPXKernelsSTD {

namespace {

template <class ExPolicy, class Scalar, class ElementType,
    std::experimental::extents<>::size_type ext0, class Layout, class Accessor>
void linalg_scale_rank_1(ExPolicy&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType, std::experimental::extents<ext0>,
        Layout, Accessor>
        x)
{
#if defined(HPX_HAVE_DATAPAR)
    using mdspan_t = std::experimental::mdspan<ElementType,
        std::experimental::extents<ext0>, Layout, Accessor>;

    constexpr bool allow_explicit_vectorization =
        mdspan_t::is_always_contiguous() &&
        (hpx::is_vectorpack_execution_policy_v<ExPolicy> ||
            hpx::is_unsequenced_execution_policy_v<ExPolicy>);

    if constexpr (allow_explicit_vectorization)
    {
        // vectorize only if the array is contiguous and not strided
        if (x.is_contiguous() && x.stride(0) == 1)
        {
            hpx::for_each(policy, x.data(), x.data() + x.extent(0),
                [&](auto& v) { v *= alpha; });
        }
        else
        {
            // fall back to the underlying base policy
            hpx::experimental::for_loop(policy.base_policy(),
                std::experimental::extents<>::size_type(0), x.extent(0),
                [&](auto i) { x(i) *= alpha; });
        }
    }
    else
#endif
    {
        hpx::experimental::for_loop(policy,
            std::experimental::extents<>::size_type(0), x.extent(0),
            [&](auto i) { x(i) *= alpha; });
    }
}

template <class ExPolicy, class Scalar, class ElementType,
    std::experimental::extents<>::size_type numRows,
    std::experimental::extents<>::size_type numCols, class Layout,
    class Accessor>
void linalg_scale_rank_2(ExPolicy&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType,
        std::experimental::extents<numRows, numCols>, Layout, Accessor>
        A)
{
    using size_type = typename std::experimental::extents<>::size_type;
    hpx::experimental::for_loop(policy,
        std::experimental::extents<>::size_type(0), A.extent(1), [&](auto j) {
            for (size_type i = 0; i < A.extent(0); ++i)
            {
                A(i, j) *= alpha;
            }
        });
}

}    // namespace

MDSPAN_TEMPLATE_REQUIRES(class ExPolicy, class Scalar, class ElementType,
    std::experimental::extents<>::size_type... ext, class Layout,
    class Accessor,
    /* requires */ (sizeof...(ext) <= 2))
void scale(hpx_exec<ExPolicy>&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType, std::experimental::extents<ext...>,
        Layout, Accessor>
        x)
{
    Impl::signal_hpx_impl_called("scale");
    if constexpr (x.rank() == 1)
    {
        linalg_scale_rank_1(policy.policy_, alpha, x);
    }
    else if constexpr (x.rank() == 2)
    {
        linalg_scale_rank_2(policy.policy_, alpha, x);
    }
}

}    // namespace HPXKernelsSTD
#endif
