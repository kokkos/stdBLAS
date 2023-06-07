
#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "dot example: calling kokkos-kernels" << std::endl;

  std::size_t N = 50;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;

    Kokkos::View<value_type*> a_view("A",N);
    Kokkos::View<value_type*> b_view("B",N);
    value_type* a_ptr = a_view.data();
    value_type* b_ptr = b_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type a(a_ptr,N);
    mdspan_type b(b_ptr,N);
    for(std::size_t i=0; i<a.extent(0); i++){
      a(i) = i;
      b(i) = i;
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value(2.0);

    // This goes to the base implementation
    const auto res_seq = stdla::dot(std::execution::seq, a, b, init_value);
    printf("Seq result    = %lf\n", res_seq);

    // This forwards to KokkosKernels
    const auto res_kk = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(), a, b, init_value);
    printf("Kokkos result = %lf\n", res_kk);
  }
  Kokkos::finalize();
}
