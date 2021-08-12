#include <experimental/linalg>

#include <iostream>

using namespace std::experimental;
int main(int argc, char* argv[]) {
  std::cout << "MatrixVectorProduct" << std::endl;
  int N = 40;
  int M = 20;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> x_view("X",M);
    Kokkos::View<double*> y_view("Y",N);
    Kokkos::View<float**,Kokkos::LayoutRight> A_view("A",N,M);

    Kokkos::deep_copy(x_view,1.0);
    Kokkos::deep_copy(A_view,2.0);

    // std::experimental::mdspan a(a_ptr,N); // Requires CDAT
    mdspan<double, extents<dynamic_extent>> x(x_view.data(),M);
    mdspan<double, extents<dynamic_extent>> y(y_view.data(),N);
    mdspan<float, extents<dynamic_extent,dynamic_extent>> A(A_view.data(),N,M);

    // This forwards to KokkosKernels (https://github.com/kokkos/kokkos-kernels
    std::experimental::linalg::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),A,x,y);
    // This forwards to KokkosKernels if LINALG_ENABLE_KOKKOS_DEFAULT is ON
    std::experimental::linalg::matrix_vector_product(A,x,y);
    std::experimental::linalg::matrix_vector_product(std::execution::par,A,linalg::scaled(2.0,x),y);
    // This goes to the base implementation
    //std::experimental::linalg::matrix_vector_product(std::execution::seq,A,x,y);

    for(int i=0; i<y.extent(0); i++) printf("%i %lf\n",i,y(i));
  }
  Kokkos::finalize();
}
