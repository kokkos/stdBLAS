
#include <iostream>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace KokkosKernelsSTD {
namespace Impl {

#if defined(KOKKOS_STDBLAS_ENABLE_TESTS)
void signal_kokkos_impl_called(std::string_view functionName)
{
  std::cout << functionName << ": kokkos impl" << std::endl;
}
#endif

} // namespace Impl
} // namespace KokkosKernelsSTD

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc,argv);
  int err = 0;
  {
    Kokkos::initialize (argc, argv);
    err = RUN_ALL_TESTS();
    Kokkos::finalize();
  }
  return err;
}
