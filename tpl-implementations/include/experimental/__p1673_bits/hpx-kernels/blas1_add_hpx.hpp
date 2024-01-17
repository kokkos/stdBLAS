/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_ADD_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_ADD_HPP_

#include <hpx/algorithm.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include <cstddef>
#include <type_traits>

#include "exec_policy_wrapper_hpx.hpp"
#include "signal_hpx_impl_called.hpp"

namespace HPXKernelsSTD {

namespace {

template <class ExPolicy, class ElementType_x, class SizeType_x,
    ::std::size_t ext_x, class Layout_x, class Accessor_x, class ElementType_y,
    class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
    class ElementType_z, class SizeType_z, ::std::size_t ext_z, class Layout_z,
    class Accessor_z>
void add_rank_1(ExPolicy&& policy,
    std::experimental::mdspan<ElementType_x,
        std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x>
        x,
    std::experimental::mdspan<ElementType_y,
        std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y>
        y,
    std::experimental::mdspan<ElementType_z,
        std::experimental::extents<SizeType_z, ext_z>, Layout_z, Accessor_z>
        z)
{
    static_assert(x.static_extent(0) == std::experimental::dynamic_extent ||
        z.static_extent(0) == std::experimental::dynamic_extent ||
        x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == std::experimental::dynamic_extent ||
        z.static_extent(0) == std::experimental::dynamic_extent ||
        y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == std::experimental::dynamic_extent ||
        y.static_extent(0) == std::experimental::dynamic_extent ||
        x.static_extent(0) == y.static_extent(0));

#if defined(HPX_HAVE_DATAPAR)
    if constexpr (supports_vectorization_v<ExPolicy>)
    {
        using mdspan_x_t = std::experimental::mdspan<ElementType_x,
            std::experimental::extents<SizeType_x, ext_x>, Layout_x,
            Accessor_x>;
        using mdspan_y_t = std::experimental::mdspan<ElementType_y,
            std::experimental::extents<SizeType_y, ext_y>, Layout_y,
            Accessor_y>;
        using mdspan_z_t = std::experimental::mdspan<ElementType_z,
            std::experimental::extents<SizeType_z, ext_z>, Layout_z,
            Accessor_z>;

        if constexpr (allow_vectorization_v<mdspan_x_t> &&
            allow_vectorization_v<mdspan_y_t> &&
            allow_vectorization_v<mdspan_z_t>)
        {
            // vectorize only if the arrays are contiguous and not strided
            if (x.is_contiguous() && x.stride(0) == 1 && y.is_contiguous() &&
                y.stride(0) == 1 && z.is_contiguous() && z.stride(0) == 1)
            {
                auto zip = hpx::util::make_zip_iterator(x.data(), y.data());
                hpx::transform(policy, zip, zip + x.extent(0), z.data(),
                    [&](auto v) { return hpx::get<0>(v) + hpx::get<1>(v); });
            }
            else
            {
                // fall back to the underlying base policy
                hpx::experimental::for_loop(
                    hpx::execution::experimental::to_non_simd(policy),
                    SizeType_z(0), x.extent(0),
                    [&](auto i) { z(i) = x(i) + y(i); });
            }
        }
        else
        {
            hpx::experimental::for_loop(
                hpx::execution::experimental::to_non_simd(policy),
                SizeType_z(0), z.extent(0),
                [&](auto i) { z(i) = x(i) + y(i); });
        }
    }
    else
#endif
    {
        hpx::experimental::for_loop(policy, SizeType_z(0), z.extent(0),
            [&](auto i) { z(i) = x(i) + y(i); });
    }
}

template <class ExPolicy, class ElementType_x, class SizeType_x,
    ::std::size_t numRows_x, ::std::size_t numCols_x, class Layout_x,
    class Accessor_x, class ElementType_y, class SizeType_y,
    ::std::size_t numRows_y, ::std::size_t numCols_y, class Layout_y,
    class Accessor_y, class ElementType_z, class SizeType_z,
    ::std::size_t numRows_z, ::std::size_t numCols_z, class Layout_z,
    class Accessor_z>
void add_rank_2(ExPolicy&& policy,
    std::experimental::mdspan<ElementType_x,
        std::experimental::extents<SizeType_x, numRows_x, numCols_x>, Layout_x,
        Accessor_x>
        x,
    std::experimental::mdspan<ElementType_y,
        std::experimental::extents<SizeType_y, numRows_y, numCols_y>, Layout_y,
        Accessor_y>
        y,
    std::experimental::mdspan<ElementType_z,
        std::experimental::extents<SizeType_z, numRows_z, numCols_z>, Layout_z,
        Accessor_z>
        z)
{
    static_assert(x.static_extent(0) == std::experimental::dynamic_extent ||
        z.static_extent(0) == std::experimental::dynamic_extent ||
        x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == std::experimental::dynamic_extent ||
        z.static_extent(0) == std::experimental::dynamic_extent ||
        y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == std::experimental::dynamic_extent ||
        y.static_extent(0) == std::experimental::dynamic_extent ||
        x.static_extent(0) == y.static_extent(0));

    static_assert(x.static_extent(1) == std::experimental::dynamic_extent ||
        z.static_extent(1) == std::experimental::dynamic_extent ||
        x.static_extent(1) == z.static_extent(1));
    static_assert(y.static_extent(1) == std::experimental::dynamic_extent ||
        z.static_extent(1) == std::experimental::dynamic_extent ||
        y.static_extent(1) == z.static_extent(1));
    static_assert(x.static_extent(1) == std::experimental::dynamic_extent ||
        y.static_extent(1) == std::experimental::dynamic_extent ||
        x.static_extent(1) == y.static_extent(1));

    hpx::experimental::for_loop(
        policy, SizeType_z(0), z.extent(1), [&](auto j) {
            for (SizeType_z i = 0; i < z.extent(0); ++i)
            {
                z(i, j) = x(i, j) + y(i, j);
            }
        });
}

}    // end anonymous namespace

MDSPAN_TEMPLATE_REQUIRES(class ExPolicy, class ElementType_x, class SizeType_x,
    ::std::size_t... ext_x, class Layout_x, class Accessor_x,
    class ElementType_y, class SizeType_y, ::std::size_t... ext_y,
    class Layout_y, class Accessor_y, class ElementType_z, class SizeType_z,
    ::std::size_t... ext_z, class Layout_z, class Accessor_z,
    /* requires */
    (sizeof...(ext_x) == sizeof...(ext_y) &&
        sizeof...(ext_x) == sizeof...(ext_z) && sizeof...(ext_z) <= 2))
void add(hpx_exec<ExPolicy>&& policy,
    std::experimental::mdspan<ElementType_x,
        std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
        x,
    std::experimental::mdspan<ElementType_y,
        std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
        y,
    std::experimental::mdspan<ElementType_z,
        std::experimental::extents<SizeType_z, ext_z...>, Layout_z, Accessor_z>
        z)
{
    if constexpr (z.rank() == 1)
    {
        add_rank_1(policy.policy_, x, y, z);
    }
    else if constexpr (z.rank() == 2)
    {
        add_rank_2(policy.policy_, x, y, z);
    }
}

}    // namespace HPXKernelsSTD

#endif    //LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_ADD_HPP_
