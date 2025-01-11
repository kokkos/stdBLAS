//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

template<class ScalingFactor, class NestedAccessor>
class scaled_accessor {
public:
  using element_type =
    std::add_const_t<decltype(std::declval<ScalingFactor>() * std::declval<typename NestedAccessor::element_type>())>;
  using reference = std::remove_const_t<element_type>;
  using data_handle_type = typename NestedAccessor::data_handle_type;
  using offset_policy =
    scaled_accessor<ScalingFactor, typename NestedAccessor::offset_policy>;

  constexpr scaled_accessor() = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherScalingFactor,
    class OtherNestedAccessor,
    /* requires */ (
      std::is_constructible_v<NestedAccessor, const OtherNestedAccessor&> &&
      std::is_constructible_v<ScalingFactor, OtherScalingFactor>
    )
  )
#if defined(__cpp_conditional_explicit)
  explicit(!std::is_convertible_v<OtherNestedAccessor, NestedAccessor>)
#endif
  constexpr scaled_accessor(const scaled_accessor<OtherScalingFactor, OtherNestedAccessor>& other) :
    scaling_factor_(other.scaling_factor()),
    nested_accessor_(other.nested_accessor())
  {}

  constexpr scaled_accessor(const ScalingFactor& s, const NestedAccessor& a) :
    scaling_factor_(s),
    nested_accessor_(a)
  {}

  constexpr reference access(data_handle_type p, ::std::size_t i) const {
    return scaling_factor_ * typename NestedAccessor::element_type(nested_accessor_.access(p, i));
  }

  typename offset_policy::data_handle_type
  constexpr offset(data_handle_type p, ::std::size_t i) const {
    return nested_accessor_.offset(p, i);
  }

  constexpr NestedAccessor nested_accessor() const noexcept {
    return nested_accessor_;
  }

  constexpr ScalingFactor scaling_factor() const noexcept {
    return scaling_factor_;
  }

private:
  ScalingFactor scaling_factor_;
  NestedAccessor nested_accessor_;
};

namespace impl {

template<class ScalingFactor,
         class NestedAccessor>
using scaled_element_type =
  std::add_const_t<typename scaled_accessor<ScalingFactor, NestedAccessor>::element_type>;

} // namespace impl

template<class ScalingFactor,
         class ElementType,
         class Extents,
         class Layout,
         class Accessor>
mdspan<impl::scaled_element_type<ScalingFactor, Accessor>,
       Extents,
       Layout,
       scaled_accessor<ScalingFactor, Accessor>>
scaled(ScalingFactor scaling_factor,
       mdspan<ElementType, Extents, Layout, Accessor> x)
{
  using acc_type = scaled_accessor<ScalingFactor, Accessor>;
  return {x.data_handle(), x.mapping(), acc_type{scaling_factor, x.accessor()}};
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
