#include <experimental/linalg>

#include <iostream>


int main(int argc, char* argv[]) {
  std::cout << "hello world" << std::endl;
  int N = 40;
#ifdef LINALG_ENABLE_KOKKOS
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> a_view("A",N);
    double* a_ptr = a_view.data();
#else
  {
    std::vector<double> a_vec(N);
    double* a_ptr = a_vec.data();
#endif

    std::experimental::mdspan a(a_ptr,N);
    for(int i=0; i<a.extent(0); i++) a(i) = i;

#ifdef LINALG_ENABLE_KOKKOS
    std::experimental::linalg::scale(KokkosKernelsSTD::kokkos_exec<>(),2.0,a);
#endif
    std::experimental::linalg::scale(std::execution::par,2.0,a);
    std::experimental::linalg::scale(std::execution::seq,2.0,a);
    for(int i=0; i<a.extent(0); i++) printf("%i %lf\n",i,a(i));
  }
#ifdef LINALG_ENABLE_KOKKOS
  Kokkos::finalize();
#endif
}
