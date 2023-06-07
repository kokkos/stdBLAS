
#include <experimental/linalg>
#include <iostream>

int main(int argc, char* argv[])
{
  std::cout << "dotc example: calling kokkos-kernels" << std::endl;

  std::size_t N = 10;
  Kokkos::initialize(argc,argv);
  {
    using value_type = std::complex<double>;
    using view_t = Kokkos::View<value_type*>;
    view_t a_view("A",N);
    view_t b_view("B",N);
    value_type* a_ptr = a_view.data();
    value_type* b_ptr = b_view.data();

    using dyn_1d_ext_type = std::experimental::extents<std::dynamic_extent>;
    using mdspan_type  = std::experimental::mdspan<value_type, dyn_1d_ext_type>;
    mdspan_type a(a_ptr,N);
    mdspan_type b(b_ptr,N);
    for(std::size_t i=0; i<a.extent(0); i++)
    {
      const auto i_double = static_cast<double>(i);
      const value_type a_i(i_double + 1.0, i_double + 1.0);
      const value_type b_i(i_double - 2.0, i_double - 2.0);
      a(i) = a_i;
      b(i) = b_i;
    }

    namespace stdla = std::experimental::linalg;
    const value_type init_value(2., 3.);

    // This goes to the base implementation
    const auto res_seq = stdla::dotc(std::execution::seq, a, b, init_value);
    std::cout << "Seq result    = " << res_seq << "\n";

    // This forwards to KokkosKernels
    const auto res_kk = stdla::dotc(KokkosKernelsSTD::kokkos_exec<>(), a, b, init_value);
    std::cout << "Kokkos result = " << res_kk << "\n";
  }
  Kokkos::finalize();
}
