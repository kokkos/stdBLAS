// For GCC 9 (< 10), <execution> unconditionally includes a TBB header file.
// If GCC < 10 was not built with TBB support, this causes a build error.
#ifdef LINALG_HAS_EXECUTION
#include <execution>
#endif

#include <iostream>
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
template<class T>
auto execpolicy_mapper(const T) {
  std::cerr << "execpolicy_mapper returning inline_exec_t\n";
  return std::experimental::linalg::impl::inline_exec_t(); 
}

#ifdef LINALG_HAS_EXECUTION
template<>
auto execpolicy_mapper(const std::execution::parallel_policy) {
  std::cerr << "execpolicy_mapper returning par\n";
  return std::execution::par;
}
#endif

}
}
}
}
