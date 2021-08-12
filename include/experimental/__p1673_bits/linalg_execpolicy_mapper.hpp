#include<execution>
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
}
}
}
}
}


#if defined(LINALG_ENABLE_KOKKOS) && defined(LINALG_ENABLE_KOKKOS_DEFAULT)
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
