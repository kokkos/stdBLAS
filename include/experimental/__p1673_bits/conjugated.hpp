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
#include <complex>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

template<class Reference, class ElementType>
class conjugated_scalar {
public:
  using value_type = ElementType;

  conjugated_scalar(Reference v) : val(v) {}

  operator ElementType() const { return conj(val); }

  template<class T2>
  auto operator* (const T2 upd) const {
    using std::conj;
    return conj(val) * upd;
  }

  template<class T2>
  auto operator+ (const T2 upd) const {
    using std::conj;
    return conj(val) + upd;
  }

  template<class T2>
  bool operator== (const T2 upd) const {
    using std::conj;
    return conj(val) == upd;
  }

  template<class T2>
  bool operator!= (const T2 upd) const {
    using std::conj;
    return conj(val) != upd;
  }

private:
  Reference val;
};

template<class T1, class Reference, class Element>
auto operator* (const T1 x, const conjugated_scalar<Reference, Element> y) {
  return x * Reference(y);
}

template<class Accessor>
class accessor_conjugate {
public:
  using element_type  = typename Accessor::element_type;

private:
  using size_type = typename extents<>::size_type;
  static constexpr bool element_type_is_complex =
    std::is_same_v<element_type, std::complex<float>> ||
    std::is_same_v<element_type, std::complex<double>> ||
    std::is_same_v<element_type, std::complex<long double>>;
  
public:
  using pointer       = typename Accessor::pointer;
  using reference     = typename std::conditional_t<
    element_type_is_complex,
    conjugated_scalar<typename Accessor::reference, element_type>,
    typename Accessor::reference>;
  using offset_policy = typename std::conditional_t<
    element_type_is_complex,
    accessor_conjugate<typename Accessor::offset_policy>,
    typename Accessor::offset_policy>;

  accessor_conjugate() = default;

  accessor_conjugate(Accessor a) : acc(a) {}

  reference access(pointer p, size_type i) const noexcept {
    return reference(acc.access(p, i));
  }

  typename offset_policy::pointer offset(pointer p, size_type i) const noexcept {
    return acc.offset(p, i);
  }

  element_type* decay(pointer p) const noexcept {
    return acc.decay(p);
  }

  // NOT IN PROPOSAL
  //
  // This isn't marked noexcept because that would impose a constraint
  // on Accessor's copy constructor.
  Accessor nested_accessor() const {
    return acc;
  }
  
private:
  Accessor acc;
};

template<class EltType, class Extents, class Layout, class Accessor>
mdspan<EltType, Extents, Layout, accessor_conjugate<Accessor>>
conjugated(mdspan<EltType, Extents, Layout, Accessor> a)
{
  using accessor_t = accessor_conjugate<Accessor>;
  return mdspan<EltType, Extents, Layout, accessor_t>{
    a.data(), a.mapping(), accessor_t(a.accessor())};
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
