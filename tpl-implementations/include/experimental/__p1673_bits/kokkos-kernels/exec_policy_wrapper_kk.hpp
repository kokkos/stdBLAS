#ifndef __LINALG_KOKKOSKERNELS_EXEC_POLICY_WRAPPER_KK_HPP_
#define __LINALG_KOKKOSKERNELS_EXEC_POLICY_WRAPPER_KK_HPP_
#include<Kokkos_Core.hpp>
#include<execution>
namespace KokkosKernelsSTD {

template<class ExecSpace = Kokkos::DefaultExecutionSpace>
struct kokkos_exec {
};

auto execpolicy_mapper(kokkos_exec<Kokkos::DefaultExecutionSpace>) { return kokkos_exec<>(); }
}

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {
  auto execpolicy_mapper(std::execution::parallel_policy) { return KokkosKernelsSTD::kokkos_exec<>(); }
  auto execpolicy_mapper(std::execution::parallel_unsequenced_policy) { return KokkosKernelsSTD::kokkos_exec<>(); }
}
}
}
}
#endif
