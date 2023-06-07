
#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "vector_norm2 example: calling kokkos-kernels" << std::endl;

  std::size_t N = 20;
  Kokkos::initialize(argc,argv);
  {
    using value_type = double;
    Kokkos::View<value_type*> x_view("x",N);
    value_type* x_ptr = x_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type x(x_ptr,N);
    for(std::size_t i=0; i<x.extent(0); i++){
      x(i) = static_cast<value_type>(i);
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value(2);

    const auto res = stdla::vector_norm2(x, init_value);
    printf("Default result    = %lf\n", res);

    const auto res_kk = stdla::vector_norm2(KokkosKernelsSTD::kokkos_exec<>(), x, init_value);
    printf("Kokkos result = %lf\n", res_kk);

  }
  Kokkos::finalize();
}
