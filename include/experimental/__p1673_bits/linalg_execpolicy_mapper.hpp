#ifdef LINALG_HAS_EXECUTION
#  include <execution>
#endif

#include <type_traits>
#include <utility>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {
// the execution policy used for default serial inline implementations
struct inline_exec_t {};

// The execution policy used when no execution policy is provided
// It must be remapped to some other execution policy, which the default mapper does
struct default_exec_t {};

// helpers
template<class T> struct is_inline_exec : std::false_type{};
template<> struct is_inline_exec<inline_exec_t> : std::true_type{};
template<class T> inline constexpr bool is_inline_exec_v = is_inline_exec<T>::value;

// value is true if and only if T is one of the std::linalg-specific
// custom execution policies provided by this implementation.
// Specialize this to be true for any new custom execution policy.
template<class T>
inline constexpr bool is_custom_linalg_execution_policy_v =
  std::is_same_v<T, default_exec_t> || std::is_same_v<T, inline_exec_t>;

// value is true if and only if T is _not_ inline_exec, and if T is
//
// * a Standard execution policy (like std::execution::parallel_policy),
// * one of the C++ implementation-specific execution policies, or
// * one of the std::linalg-specific custom execution policies
//   (other than inline_exec).
//
// This helps disambiguate ExecutionPolicy from otherwise
// unconstrained template parameters like ScaleFactorType in
// algorithms like symmetric_matrix_rank_k_update.
template<class T>
inline constexpr bool is_linalg_execution_policy_other_than_inline_v =
  ! is_inline_exec_v<T> &&
  (
#ifdef LINALG_HAS_EXECUTION
    std::is_execution_policy_v<T> ||
#endif
    is_custom_linalg_execution_policy_v<T>
  );

} // namespace impl
} // namespace linalg
} // inline namespace __p1673_version_0
} // namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#if defined(LINALG_ENABLE_KOKKOS)
#include <experimental/__p1673_bits/kokkos-kernels/exec_policy_wrapper_kk.hpp>
#endif


namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// Specialize this function to map a public execution policy
// (e.g., std::execution::parallel_policy) to an internal policy.
// This function must always return a different type than its input.
template<class T>
auto execpolicy_mapper(T) { return impl::inline_exec_t(); }

namespace impl {

// std::remove_cvref_t is a C++20 feature.
template<class T>
using remove_cvref_t =
#ifdef __cpp_lib_remove_cvref
  std::remove_cvref_t<T>;
#else
  std::remove_const_t<std::remove_reference_t<decltype(policy)>>;
#endif

// This function is not to be specialized; that's why
// it's a generic lambda instead of a function template.
inline auto map_execpolicy_with_check = [](auto&& policy) {
  using input_type = remove_cvref_t<decltype(policy)>;
  using return_type = remove_cvref_t<decltype(execpolicy_mapper(std::forward<decltype(policy)>(policy)))>;
  // Only inline_exec_t is allowed to map to itself.
  using inline_type = impl::inline_exec_t;
  static_assert(std::is_same_v<input_type, inline_type> ||
    ! std::is_same_v<input_type, return_type>,
    "Specializations of execpolicy_mapper must return "
    "a different policy type than their input");
  return execpolicy_mapper(std::forward<decltype(policy)>(policy));
};

} // namespace impl

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std
