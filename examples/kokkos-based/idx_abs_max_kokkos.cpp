#include <experimental/linalg>
#include <iostream>

namespace stdexp = std::experimental;
namespace stdla = stdexp::linalg;
using value_type = double;

void run_trivial_example()
{
  std::array<double, 0> arr;
  using extents_type = stdexp::extents<stdexp::dynamic_extent>;
  stdexp::mdspan<value_type, extents_type> a(arr.data(),0);

  const auto idx = stdla::vector_idx_abs_max(std::execution::seq, a);
  std::cout << "Sequen result = " << idx << '\n';

  const auto idx_kk = stdla::vector_idx_abs_max(KokkosKernelsSTD::kokkos_exec<>(), a);
  std::cout << "Kokkos result = " << idx_kk << '\n';
}

void run_nontrivial_example()
{
  std::size_t N = 10;

  Kokkos::View<value_type*> a_view("A",N);
  value_type* a_ptr = a_view.data();

  using extents_type = stdexp::extents<stdexp::dynamic_extent>;
  stdexp::mdspan<value_type, extents_type> a(a_ptr,N);
  a(0) =  0.5;
  a(1) =  0.2;
  a(2) =  0.1;
  a(3) =  0.4;
  a(4) = -0.8;
  a(5) = -1.7;
  a(6) = -0.3;
  a(7) =  0.5;
  a(8) = -1.7;
  a(9) = -0.9;

  // This goes to the base implementation
  const auto idx = stdla::vector_idx_abs_max(std::execution::seq, a);
  std::cout << "Sequen result = " << idx << '\n';

  // This forwards to KokkosKernels (https://github.com/kokkos/kokkos-kernels
  const auto idx_kk = stdla::vector_idx_abs_max(KokkosKernelsSTD::kokkos_exec<>(), a);
  std::cout << "Kokkos result = " << idx_kk << '\n';
}

int main(int argc, char* argv[])
{
  std::cout << "vector_idx_abs_max example: calling kokkos-kernels" << std::endl;

  Kokkos::initialize(argc,argv);
  {
    run_trivial_example();
    run_nontrivial_example();
  }
  Kokkos::finalize();
}
