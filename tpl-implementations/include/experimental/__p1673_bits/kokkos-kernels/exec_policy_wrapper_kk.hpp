#ifndef __LINALG_KOKKOSKERNELS_EXEC_POLICY_WRAPPER_KK_HPP_
#define __LINALG_KOKKOSKERNELS_EXEC_POLICY_WRAPPER_KK_HPP_
#include<Kokkos_Core.hpp>
#include<execution>
namespace KokkosKernelsSTD {

template<class ExecSpace = Kokkos::DefaultExecutionSpace>
struct kokkos_exec {
};

template<class ExecSpace>
auto execpolicy_mapper(kokkos_exec<ExecSpace>) { return kokkos_exec<ExecSpace>(); }
} // namespace KokkosKernelsSTD

// Remap standard execution policies to Kokkos
#ifdef LINALG_ENABLE_KOKKOS_DEFAULT
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
  auto execpolicy_mapper(impl::default_exec_t) { return KokkosKernelsSTD::kokkos_exec<>(); }
  auto execpolicy_mapper(std::execution::parallel_policy) { return KokkosKernelsSTD::kokkos_exec<>(); }
  auto execpolicy_mapper(std::execution::parallel_unsequenced_policy) { return KokkosKernelsSTD::kokkos_exec<>(); }
}
}
}
}
#endif
#endif
