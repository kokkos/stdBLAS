#include <experimental/linalg>

#include <iostream>


int main(int argc, char* argv[]) {
  std::cout << "hello world" << std::endl;
  int N = 40;
  {
    std::vector<double> a_vec(N);
    double* a_ptr = a_vec.data();

    // If CDAT works correctly this could be mdspan a(a_ptr,N);
    std::experimental::mdspan<double,std::experimental::extents<std::experimental::dynamic_extent>> a(a_ptr,N);
    for(int i=0; i<a.extent(0); i++) a(i) = i;

    std::experimental::linalg::scale(2.0,a);
    std::experimental::linalg::scale(std::execution::par, 2.0,a);
    for(int i=0; i<a.extent(0); i++) printf("%i %lf\n",i,a(i));
  }
}

