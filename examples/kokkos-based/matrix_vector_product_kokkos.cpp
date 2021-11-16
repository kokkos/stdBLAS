#include <experimental/linalg>

#include <iostream>

using namespace std::experimental;

int main(int argc, char* argv[]) {
  std::cout << "MatrixVectorProduct" << std::endl;
  int N = 10;
  int M = 20;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> x_view("X",M);
    Kokkos::View<double*> y_view("Y",N);
    Kokkos::View<float**,Kokkos::LayoutRight> A_view("A",N,M);

    {
      // example for y = A * x

      Kokkos::deep_copy(x_view,1.0);
      Kokkos::deep_copy(A_view,2.0);

      // std::experimental::mdspan a(a_ptr,N); // Requires CDAT
      mdspan<double, extents<dynamic_extent>> x(x_view.data(),M);
      mdspan<double, extents<dynamic_extent>> y(y_view.data(),N);
      mdspan<float, extents<dynamic_extent,dynamic_extent>> A(A_view.data(),N,M);

      // This forwards to KokkosKernels (https://github.com/kokkos/kokkos-kernels
      linalg::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),A,x,y);
      // This forwards to KokkosKernels if LINALG_ENABLE_KOKKOS_DEFAULT is ON
      linalg::matrix_vector_product(A,x,y);
      linalg::matrix_vector_product(std::execution::par,A,linalg::scaled(2.0,x),y);
      // This goes to the base implementation
      //linalg::matrix_vector_product(std::execution::seq,A,x,y);

      // note that this prints 80 for each element because of the scale(2.0, x) above
      for(int i=0; i<y.extent(0); i++) printf("%i %lf, expected = %lf\n",i,y(i), 80.);
    }

    {
      // example for z = y + A * x

      Kokkos::View<double*> z_view("Z",N);
      Kokkos::deep_copy(x_view,1.0);
      Kokkos::deep_copy(y_view,2.0);
      Kokkos::deep_copy(z_view,0.0);
      Kokkos::deep_copy(A_view,1.0);

      mdspan<double, extents<dynamic_extent>> x(x_view.data(),M);
      mdspan<double, extents<dynamic_extent>> y(y_view.data(),N);
      mdspan<double, extents<dynamic_extent>> z(z_view.data(),N);
      mdspan<float, extents<dynamic_extent,dynamic_extent>> A(A_view.data(),N,M);

      // 1.
      linalg::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(),A,x,y,z);
      // should print 22
      for(int i=0; i<y.extent(0); i++) printf("%i %lf, expected = %lf\n",i,z(i), 22.);

      // 2.
      // scale y by 4 when passing it to kernel
      linalg::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(), A, x, linalg::scaled(4.,y), z);
      // should print 28
      for(int i=0; i<y.extent(0); i++) printf("%i %lf, expected = %lf\n",i,z(i), 28.);

      // 3.
      // scale y by 4 and x by 2 when passing it to kernel
      linalg::matrix_vector_product(KokkosKernelsSTD::kokkos_exec<>(), A, linalg::scaled(2., x), linalg::scaled(4.,y), z);
      // should print 48
      for(int i=0; i<y.extent(0); i++) printf("%i %lf, expected = %lf\n",i,z(i), 48.);
    }

  }
  Kokkos::finalize();
}
