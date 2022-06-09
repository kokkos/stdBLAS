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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_

#include <experimental/mdspan>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

template<class ScalingFactor, class Accessor>
class accessor_scaled {
public:
  using reference     =
    scaled_scalar<ScalingFactor, typename Accessor::reference,
      std::remove_cv_t<typename Accessor::element_type>>;
  using element_type  = std::add_const_t<typename reference::value_type>;
  using pointer       = typename Accessor::pointer;
  using offset_policy =
    accessor_scaled<ScalingFactor, typename Accessor::offset_policy>;

  accessor_scaled(ScalingFactor scaling_factor, Accessor accessor) :
    scaling_factor_(std::move(scaling_factor)), accessor_(accessor)
  {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (std::is_convertible_v<
      typename default_accessor<OtherElementType>::element_type(*)[],
      typename Accessor::element_type(*)[]
    >)
  )
  accessor_scaled(ScalingFactor scaling_factor,
		  default_accessor<OtherElementType> accessor) :
    scaling_factor_(std::move(scaling_factor)), accessor_(accessor)
  {}
  
  reference access(pointer p, extents<>::size_type i) const noexcept {
    return reference(scaling_factor_, accessor_.access(p, i));
  }

  typename offset_policy::pointer
  offset(pointer p, extents<>::size_type i) const noexcept {
    return accessor_.offset(p, i);
  }

  // NOT IN PROPOSAL
  //
  // This isn't marked noexcept because that would impose a constraint
  // on Accessor's copy constructor.
  Accessor nested_accessor() const {
    return accessor_;
  }

  ScalingFactor scaling_factor() const {
    return scaling_factor_;
  }

private:
  ScalingFactor scaling_factor_;
  Accessor accessor_;
};

namespace impl {

template<class ScalingFactor,
         class Accessor>
using scaled_element_type =
  typename accessor_scaled<ScalingFactor, Accessor>::element_type;

template<class ScalingFactor,
         class Accessor>
using scaled_accessor_type = accessor_scaled<ScalingFactor, Accessor>;

// FIXME (mfh 2022/06/08) Nested scaled applications need to preserve
// the type of the nonlazy operation.  This means preserving the
// original multiplication order and not reparenthesizing, unless they
// can prove that the type would be the same regardless.

} // namespace impl

// FIXME (mfh 2022/06/08) Spec is wrong here,
// because it doesn't preserve the type.
  
template<class ScalingFactor,
         class ElementType,
         class Extents,
         class Layout,
         class Accessor>
mdspan<impl::scaled_element_type<ScalingFactor, Accessor>,
       Extents,
       Layout,
       impl::scaled_accessor_type<ScalingFactor, Accessor>>
scaled(ScalingFactor scaling_factor,
       mdspan<ElementType, Extents, Layout, Accessor> a)
{
  using return_accessor_type = accessor_scaled<ScalingFactor, Accessor>;
  return {a.data(), a.mapping(),
	  return_accessor_type{scaling_factor, a.accessor()}};
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
