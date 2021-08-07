#include<execution>
#if defined(LINALG_ENABLE_KOKKOS) && defined(LINALG_ENABLE_KOKKOS_DEFAULT)
#include <exec_policy_wrapper_kk.hpp>
#endif
namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {
template<class T>
auto execpolicy_mapper(T) { return std::execution::seq; }
}
}
}
}
