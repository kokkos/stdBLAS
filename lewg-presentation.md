# Free function BLAS-based Linear Algebra

## Background: BLAS Standard

* Existing standard (not ISO) developed in the 90s 
* Widely used in Science and Engineering code
* Many hardware vendors provide optimized implementations as part of system libraries
  * Intel: Math Kernel Library (MKL) 
  * NVIDIA: CUBLAS
  * IBM: Engineering and Scientific Subroutine Library (ESSL)
  * ARM: Arm Performance Libraries
  * HPE: Cray LibSCI
  
* BUT: it is in Fortran, maybe with a C interface
  
## Example BLAS Matrix Vector Multiply

```c++
// Matrix
int nRows = ...;
int nCols = ...;
double* A = new double[nRows*nCols];
// Vectors
double* y = new double[nRows];
double* x = new double[nCols];

// y = 1.0*A*x;
dgemv('N',nRows,nCols,1.0,A,nRows,x,1,0.0,y,1);
```

Lets upack the 11 !! parameters to compute `y = A*x`:
  * `N`: the matrix is not transposed
  * `nRows`: Number of Rows (also length of `y`)
  * `nCols`: Number of Columns (also length of `x`)
  * `1.0`: scaling factor for `A`
  * `A`: pointer to the matrix values
  * `nRows`: stride of the rows of `A` (in our case the stride is number of rows)
  * `x`: right hand side vector
  * `1`: stride of `x`
  * `1.0`: scaling of `y`
  * `y`: left hand side vector
  * `1`: stride of `y`
  
## Reasons why this is bad

* 11 !! parameters to do `y = A*x`
* hopefully you defined the extern C functor `dgemv` correct to get the fortran link right
* The type of the scalars is part of the function name (`dgemv` == `double`, `sgemv` == `float`)
   * no mixed support, e.g. `A` and `x` are `float` and `y` is `double`
* No compile time size propagation
  * Machine Learning is all about small operations with compile time known sizes
* Storage Layout of matrix `A` better be Fortran layout

## The Philosophy Behind P1673

* Linear Algebra Functions are just another set of algorithms
  * *Propose a free function interface similar to standard algorithms*
* Encapsule data in the minimal fundamental representation of multi dimensional arrays still providing necessary felxibility
  * *Use `mdspan` as parameters*
* Ensure that all functionality of the BLAS standard is covered
  * *Propose equivalent of every function in BLAS*
* Ensure that new areas of linear algebra such as Machine Learning are covered
  * *Turns out mdspan gives that to us naturally*
  
## Example `y=A*x` with P1673

```c++
// Matrix
using matrix_ext_t = extents<dynamic_extent,dynamic_extent>;
mdspan<double,dynamic_extent,dynamic_extent> A(A_ptr,nRows,nCols);
mdspan<double,dynamic_extent> x(x_ptr,nCols), y(y_ptr,nRows);

matrix_vector_multiply(y,A,x);
```

