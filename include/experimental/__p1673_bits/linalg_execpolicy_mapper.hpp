#ifdef LINALG_HAS_EXECUTION
#  include <execution>
#endif

#include <type_traits>
#include <utility>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {
// the execution policy used for default serial inline implementations
struct inline_exec_t {};

// The execution policy used when no execution policy is provided
// It must be remapped to some other execution policy, which the default mapper does
struct default_exec_t {};

#ifdef LINALG_HAS_EXECUTION
// Result of execpolicy_mapper(std::execution::par)
struct parallel_exec_t {};
#endif

// helpers
template<class T> struct is_inline_exec : std::false_type{};
template<> struct is_inline_exec<inline_exec_t> : std::true_type{};
template<class T> inline constexpr bool is_inline_exec_v = is_inline_exec<T>::value;
}
}
}
}
}


#if defined(LINALG_ENABLE_KOKKOS)
#include <experimental/__p1673_bits/kokkos-kernels/exec_policy_wrapper_kk.hpp>
#endif


namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

// Specialize this function to map a public execution policy
// (e.g., std::execution::parallel_policy) to an internal policy.
// This function must always return a different type than its input.
template<class T>
auto execpolicy_mapper(T) {
  return std::experimental::linalg::impl::inline_exec_t();
}

#ifdef LINALG_HAS_EXECUTION
// Result of execpolicy_mapper(std::execution::par)
inline auto execpolicy_mapper(const std::execution::parallel_policy&) {
  return std::experimental::linalg::impl::parallel_exec_t();
}
#endif

namespace detail {

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
  using ::std::experimental::linalg::execpolicy_mapper;
  using return_type = remove_cvref_t<decltype(execpolicy_mapper(std::forward<decltype(policy)>(policy)))>;
  // Only inline_exec_t is allowed to map to itself.
  using inline_type = ::std::experimental::linalg::impl::inline_exec_t;
  static_assert(std::is_same_v<input_type, inline_type> ||
    ! std::is_same_v<input_type, return_type>,
    "Specializations of execpolicy_mapper must return "
    "a different policy type than their input");
  return execpolicy_mapper(std::forward<decltype(policy)>(policy));
};

} // namespace detail

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std
