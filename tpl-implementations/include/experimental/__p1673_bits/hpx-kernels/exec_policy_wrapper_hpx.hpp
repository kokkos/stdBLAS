//  Copyright (c) 2022 Hartmut Kaiser

#ifndef __LINALG_HPXKERNELS_EXEC_POLICY_WRAPPER_HPX_HPP_
#define __LINALG_HPXKERNELS_EXEC_POLICY_WRAPPER_HPX_HPP_

#include <hpx/execution.hpp>

#ifdef LINALG_ENABLE_HPX_DEFAULT
#include <execution>
#endif
#include <type_traits>
#include <utility>

namespace HPXKernelsSTD {

template <class ExPolicy = hpx::execution::sequenced_policy>
struct hpx_exec
{
    using type = ExPolicy;
    ExPolicy policy_;
};

template <class ExPolicy>
auto execpolicy_mapper(hpx_exec<ExPolicy> policy)
{
    return policy;
}

template <typename MdSpan, typename Enable = void>
struct allow_vectorization : std::false_type
{};

#if defined(HPX_WITH_DATAPAR)
template <typename MdSpan>
struct allow_vectorization<MdSpan,
    std::enable_if_t<std::is_arithmetic_v<typename MdSpan::element_type>>>
  : std::true_type
{};
#endif

template <typename MdSpan>
inline constexpr bool allow_vectorization_v =
    allow_vectorization<MdSpan>::value;

template <typename ExPolicy>
inline constexpr bool supports_vectorization_v =
    hpx::is_vectorpack_execution_policy_v<ExPolicy> ||
    hpx::is_unsequenced_execution_policy_v<ExPolicy>;

}    // namespace HPXKernelsSTD

// Remap standard execution policies to HPX
#ifdef LINALG_ENABLE_HPX_DEFAULT
namespace std { namespace experimental { inline namespace __p1673_version_0 {
namespace linalg {
auto execpolicy_mapper(std::execution::parallel_policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::parallel_policy>{
        hpx::execution::par};
}
auto execpolicy_mapper(std::execution::parallel_unsequenced_policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::parallel_unsequenced_policy>{
        hpx::execution::par_unseq};
}
}}}}    // namespace std::experimental::__p1673_version_0::linalg
#endif

namespace std { namespace experimental { inline namespace __p1673_version_0 {
namespace linalg {
auto execpolicy_mapper(hpx::execution::parallel_policy policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::parallel_policy>{
        std::move(policy)};
}
auto execpolicy_mapper(hpx::execution::parallel_unsequenced_policy policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::parallel_unsequenced_policy>{
        std::move(policy)};
}
#if defined(HPX_HAVE_DATAPAR)
auto execpolicy_mapper(hpx::execution::simd_policy policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::simd_policy>{
        std::move(policy)};
}
auto execpolicy_mapper(hpx::execution::par_simd_policy policy)
{
    return HPXKernelsSTD::hpx_exec<hpx::execution::par_simd_policy>{
        std::move(policy)};
}
#endif
}}}}    // namespace std::experimental::__p1673_version_0::linalg
#endif
