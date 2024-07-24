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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

template<class NestedAccessor>
class conjugated_accessor {
private:
  using nested_element_type = typename NestedAccessor::element_type;
  using nc_result_type = decltype(impl::conj_if_needed(std::declval<nested_element_type>()));
public:
  using element_type = std::add_const_t<nc_result_type>;
  using reference = std::remove_const_t<element_type>;
  using data_handle_type = typename NestedAccessor::data_handle_type;
  using offset_policy =
    conjugated_accessor<typename NestedAccessor::offset_policy>;

  constexpr conjugated_accessor() = default;
  constexpr conjugated_accessor(const NestedAccessor& acc) : nested_accessor_(acc) {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherNestedAccessor,
    /* requires */ (std::is_convertible_v<NestedAccessor, const OtherNestedAccessor&>)
  )
  constexpr conjugated_accessor(const conjugated_accessor<OtherNestedAccessor>& other)
    : nested_accessor_(other.nested_accessor())
  {}

  constexpr reference
    access(data_handle_type p, ::std::size_t i) const noexcept
  {
    return impl::conj_if_needed(nested_element_type(nested_accessor_.access(p, i)));
  }

  constexpr typename offset_policy::data_handle_type
    offset(data_handle_type p, ::std::size_t i) const noexcept
  {
    return nested_accessor_.offset(p, i);
  }

  const NestedAccessor& nested_accessor() const noexcept { return nested_accessor_; }

private:
  NestedAccessor nested_accessor_;
};

template<class ElementType, class Extents, class Layout, class Accessor>
auto conjugated(mdspan<ElementType, Extents, Layout, Accessor> a)
{
  using value_type = typename decltype(a)::value_type;

  if constexpr (std::is_arithmetic_v<value_type>) {
    return mdspan<ElementType, Extents, Layout, Accessor>
      (a.data_handle(), a.mapping(), a.accessor());
  }
  // P3050 optimizes conjugated for nonarithmetic, non-(custom complex) types.
  // A "custom complex" type T has ADL-findable conj(T).
  else if constexpr (! impl::has_conj<value_type>::value) {
    return mdspan<ElementType, Extents, Layout, Accessor>
      (a.data_handle(), a.mapping(), a.accessor());
  }
  else {
    using return_element_type =
      typename conjugated_accessor<Accessor>::element_type;
    using return_accessor_type = conjugated_accessor<Accessor>;
    return mdspan<return_element_type, Extents, Layout, return_accessor_type>
      (a.data_handle(), a.mapping(), return_accessor_type(a.accessor()));
  }
}

// Conjugation is self-annihilating
template<class ElementType, class Extents, class Layout, class NestedAccessor>
auto conjugated(
  mdspan<ElementType, Extents, Layout, conjugated_accessor<NestedAccessor>> a)
{
  using return_element_type = typename NestedAccessor::element_type;
  using return_accessor_type = NestedAccessor;
  return mdspan<return_element_type, Extents, Layout, return_accessor_type>
    (a.data_handle(), a.mapping(), a.nested_accessor());
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
