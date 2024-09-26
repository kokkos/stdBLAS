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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_

#include <cstdlib>
#include <cmath>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace
{
template <class Exec, class v_t, class Scalar, class = void>
struct is_custom_vector_abs_sum_avail : std::false_type {};

template <class Exec, class v_t, class Scalar>
struct is_custom_vector_abs_sum_avail<
  Exec, v_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(vector_abs_sum(std::declval<Exec>(),
			      std::declval<v_t>(),
			      std::declval<Scalar>())
	       ),
      Scalar
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_abs_sum(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  using std::abs;
  using result_t = decltype(init + abs(impl::real_part(std::declval<ElementType>())));
  const SizeType numElt = v.extent(0);
  result_t value = init;
  for (SizeType i = 0; i < numElt; ++i) {
    if constexpr (std::is_arithmetic_v<ElementType>) {
        value += abs(v(i));
    }
    else {
        value += abs(impl::real_part(v(i)));
        value += abs(impl::imag_part(v(i)));
    }
  }
  return value;
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_abs_sum(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  constexpr bool use_custom = is_custom_vector_abs_sum_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(v), Scalar
    >::value;

  if constexpr (use_custom) {
    return vector_abs_sum(impl::map_execpolicy_with_check(exec), v, init);
  }
  else {
    return vector_abs_sum(impl::inline_exec_t{}, v, init);
  }
}

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_abs_sum(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  return vector_abs_sum(impl::default_exec_t{}, v, init);
}

namespace vector_abs_detail {
  using std::abs;

  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  template<
    class ElementType,
    class SizeType, ::std::size_t ext0,
    class Layout,
    class Accessor>
  auto vector_abs_return_type_deducer(
    mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
  -> decltype(abs(x(0)));
} // namespace vector_abs_detail


template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_abs_sum(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_abs_detail::vector_abs_return_type_deducer(x))
{
  using return_t = decltype(vector_abs_detail::vector_abs_return_type_deducer(x));
  return vector_abs_sum(x, return_t{});
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_abs_sum(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_abs_detail::vector_abs_return_type_deducer(x))
{
  using return_t = decltype(vector_abs_detail::vector_abs_return_type_deducer(x));
  return vector_abs_sum(exec, x, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
