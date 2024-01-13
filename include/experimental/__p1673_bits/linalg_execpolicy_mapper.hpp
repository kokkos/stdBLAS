// For GCC 9 (< 10), <execution> unconditionally includes a TBB header file.
// If GCC < 10 was not built with TBB support, this causes a build error.
#if (! defined(__GNUC__)) || (__GNUC__ > 9)
#include <execution>
#endif

#include <type_traits>

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
#if (! defined(__GNUC__)) || (__GNUC__ > 9)
    std::is_execution_policy_v<T> ||
#endif
    is_custom_linalg_execution_policy_v<T>
  );

} // namespace impl
} // namespace linalg
} // inline namespace __p1673_version_0
} // namespace experimental
} // namespace std

#if defined(LINALG_ENABLE_KOKKOS)
#include <experimental/__p1673_bits/kokkos-kernels/exec_policy_wrapper_kk.hpp>
#endif


namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {
template<class T>
auto execpolicy_mapper(T) { return std::experimental::linalg::impl::inline_exec_t(); }
}
}
}
}
