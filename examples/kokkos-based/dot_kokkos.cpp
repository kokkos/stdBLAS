
#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "running dot example calling custom kokkos" << std::endl;
  int N = 50;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> a_view("A",N);
    Kokkos::View<double*> b_view("B",N);
    double* a_ptr = a_view.data();
    double* b_ptr = b_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::experimental::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<double, dyn_1d_ext_type>;
    mdspan_type a(a_ptr,N);
    mdspan_type b(b_ptr,N);
    for(int i=0; i<a.extent(0); i++){
      a(i) = i;
      b(i) = i;
    }

    namespace stdla = std::experimental::linalg;
    const double init_value = 2.0;

    // This goes to the base implementation
    const auto res_seq = stdla::dot(std::execution::seq, a, b, init_value);

    // This forwards to KokkosKernels
    auto res_kk = stdla::dot(KokkosKernelsSTD::kokkos_exec<>(), a, b, init_value);

    printf("Kokkos result = %lf\n", res_kk);
    printf("Seq result    = %lf\n", res_seq);
  }
  Kokkos::finalize();
}
