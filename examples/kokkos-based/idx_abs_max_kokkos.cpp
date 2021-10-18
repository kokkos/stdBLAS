#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "idx_abs_max example: calling kokkos-kernels" << std::endl;

  std::size_t N = 10;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;

    Kokkos::View<value_type*> a_view("A",N);
    value_type* a_ptr = a_view.data();

    // Requires CTAD working, GCC 11.1 works but some others are buggy
    // std::experimental::mdspan a(a_ptr,N);
    using extents_type = std::experimental::extents<std::experimental::dynamic_extent>;
    std::experimental::mdspan<value_type, extents_type> a(a_ptr,N);
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

    namespace stdla = std::experimental::linalg;

    // This goes to the base implementation
    const auto idx = stdla::idx_abs_max(std::execution::seq, a);
    printf("Seq result    = %i\n", idx);

    // This forwards to KokkosKernels (https://github.com/kokkos/kokkos-kernels
    const auto idx_kk = stdla::idx_abs_max(KokkosKernelsSTD::kokkos_exec<>(), a);
    printf("Kokkos result = %i\n", idx_kk);
  }
  Kokkos::finalize();
}
