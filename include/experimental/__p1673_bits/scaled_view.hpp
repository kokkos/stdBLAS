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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_VIEW_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_VIEW_HPP_

#include <experimental/mdspan>
#include <experimental/mdarray>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

template<class T, class S>
class scaled_scalar {
public:
  scaled_scalar(const T& v, const S& s) :
    val(v), scale(s) {}

  operator T() const { return val * scale; }

  T operator- () const { return -(val * scale); }

  template<class T2>
  decltype(auto) operator+ (const T2& upd) const {
    return val*scale + upd;
  }

  template<class T2>
  decltype(auto) operator* (const T2 upd) const {
    return val*scale * upd;
  }

  // ... add only those operators needed for the functions
  // in this proposal ...

private:
  const T& val;
  const S scale;
};

template<class Accessor, class S>
class accessor_scaled {
public:
  using element_type  = typename Accessor::element_type;
  using pointer       = typename Accessor::pointer;
  using reference     = scaled_scalar<typename Accessor::reference, S>;
  using offset_policy = accessor_scaled<typename Accessor::offset_policy, S>;

  accessor_scaled(Accessor a, S sval) :
    acc(a), scale_factor(sval) {}

  reference access(pointer& p, ptrdiff_t i) const noexcept {
    return reference(acc.access(p,i), scale_factor);
  }

  typename offset_policy::pointer offset(pointer p, ptrdiff_t i) const noexcept {
    return acc.offset(p,i);
  }

  element_type* decay(pointer p) const noexcept {
    return acc.decay(p);
  }

private:
  Accessor acc;
  S scale_factor;
};

// FIXME Finish these (see e.g., "see-below")
#if 0
template<class T, class Extents, class Layout,
         class Accessor, class S>
basic_mdspan<T, Extents, Layout, accessor_scaled<Accessor, S>>
scaled_view(S s, const basic_mdspan<T, Extents, Layout, Accessor>& a);

template<class T, class Extents, class Layout,
         class Accessor, class S>
basic_mdspan<const T, Extents, Layout, <i>see-below</i> >
scaled_view(S s, const basic_mdarray<T, Extents, Layout, Accessor>& a);
#endif // 0

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_VIEW_HPP_
