
#include <experimental/linalg>
#include <iostream>

template<class T1, class ScalarType>
void print_elements(const T1 & v, const std::vector<ScalarType> & gold)
{
  for(int i=0; i<v.size(); i++){
    std::cout << "computed = " << v(i)
	      << " , gold = "
	      << gold[i]
	      << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::cout << "running add example calling custom kokkos" << std::endl;
  int N = 50;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double*> x_view("x",N);
    Kokkos::View<double*> y_view("y",N);
    Kokkos::View<double*> z_view("z",N);

    double* x_ptr = x_view.data();
    double* y_ptr = y_view.data();
    double* z_ptr = z_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::experimental::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<double, dyn_1d_ext_type>;
    mdspan_type x(x_ptr,N);
    mdspan_type y(y_ptr,N);
    mdspan_type z(z_ptr,N);

    std::vector<double> gold(N);
    for(int i=0; i<x.extent(0); i++){
      x(i) = i;
      y(i) = i + (double)10;
      z(i) = 0;
      gold[i] = x(i) + y(i);
    }

    namespace stdla = std::experimental::linalg;
    const double init_value = 2.0;

    {
      // This goes to the base implementation
      stdla::add(std::execution::seq, x, y, z);
    }

    {
      // reset z since it is modified above
      for(int i=0; i<z.extent(0); i++){ z(i) = 0; }

      // This forwards to KokkosKernels
      stdla::add(KokkosKernelsSTD::kokkos_exec<>(), x,y,z);
      print_elements(z, gold);
    }

  }
  Kokkos::finalize();
}
