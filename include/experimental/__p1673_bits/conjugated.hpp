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

#include <experimental/mdspan>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

template<class Accessor>
class conjugate_accessor {
private:
  Accessor accessor_;

public:
  using element_type = decltype(impl::conj_if_needed(std::declval<typename Accessor::element_type>()));
  using reference = element_type;
  using data_handle_type = typename Accessor::data_handle_type;
  using offset_policy = conjugate_accessor<typename Accessor::offset_policy>;

  conjugate_accessor(const Accessor& accessor) :
    accessor_(accessor)
  {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherNestedAccessor,
    /* requires */ (
      std::is_constructible_v<Accessor, const OtherNestedAccessor&>
    )
  )
  conjugate_accessor(const conjugate_accessor<OtherNestedAccessor>& other) :
    accessor_(other.nested_accessor())
  {}

  reference access(data_handle_type p, ::std::size_t i) const
    noexcept(noexcept(impl::conj_if_needed(typename Accessor::element_type(accessor_.access(p, i)))))
  {
    return impl::conj_if_needed(typename Accessor::element_type(accessor_.access(p, i)));
  }

  typename offset_policy::data_handle_type offset(data_handle_type p, ::std::size_t i) const
    noexcept(noexcept(accessor_.offset(p, i)))
  {
    return accessor_.offset(p, i);
  }

  Accessor nested_accessor() const { return accessor_; }
};

template<class ElementType, class Extents, class Layout, class Accessor>
auto conjugated(mdspan<ElementType, Extents, Layout, Accessor> a)
{
  if constexpr (std::is_arithmetic_v<std::remove_cv_t<ElementType>>) {
    return mdspan<ElementType, Extents, Layout, Accessor>
      (a.data_handle(), a.mapping(), a.accessor());
  } else {
    using return_element_type =
      typename conjugate_accessor<Accessor>::element_type;
    using return_accessor_type = conjugate_accessor<Accessor>;
    return mdspan<return_element_type, Extents, Layout, return_accessor_type>
      (a.data_handle(), a.mapping(), return_accessor_type(a.accessor()));
  }
}

// Conjugation is self-annihilating
template<class ElementType, class Extents, class Layout, class NestedAccessor>
auto conjugated(
  mdspan<ElementType, Extents, Layout, conjugate_accessor<NestedAccessor>> a)
{
  using return_element_type = typename NestedAccessor::element_type;
  using return_accessor_type = NestedAccessor;
  return mdspan<return_element_type, Extents, Layout, return_accessor_type>
    (a.data_handle(), a.mapping(), a.nested_accessor());
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
