# Free function BLAS-based Linear Algebra

## Authors

* Mark Hoemmen (mhoemme@sandia.gov) (Sandia National Laboratories)
* David Hollman (dshollm@sandia.gov) (Sandia National Laboratories)
* Christian Trott (crtrott@sandia.gov) (Sandia National Laboratories)
* Daniel Sunderland (dsunder@sandia.gov) (Sandia National Laboratories)
* Nevin Liber (nliber@anl.gov) (Argonne National Laboratory)
* Siva Rajamanickam (srajama@sandia.gov) (Sandia National Laboratories)
* Li-Ta Lo (ollie@lanl.gov) (Los Alamos National Laboratory)
* Damien Lebrun-Grandie (lebrungrandt@ornl.gov) (Oak Ridge National Laboratory)
* Graham Lopez (lopezmg@ornl.gov) (Oak Ridge National Laboratory)
* Peter Caday (peter.caday@intel.com) (Intel)
* Sarah Knepper (sarah.knepper@intel.com) (Intel)
* Piotr Luszczek (luszczek@icl.utk.edu) (University of Tennessee)
* Timothy Costa (tcosta@nvidia.com) (NVIDIA)

## Contributors

* Chip Freitag (chip.freitag@amd.com) (AMD)
* Bryce Lelbach (blelbach@nvidia.com) (NVIDIA)
* Srinath Vadlamani (Srinath.Vadlamani@arm.com) (ARM)
* Rene Vanoostrum (Rene.Vanoostrum@amd.com) (AMD)

## Background/Aims/Philosophy

### Background: BLAS Standard

* Existing standard (not ISO) developed in the 90s 
* Widely used in Science and Engineering code
* Many hardware vendors provide optimized implementations as part of system libraries
  * Intel: Math Kernel Library (MKL) 
  * NVIDIA: CUBLAS
  * IBM: Engineering and Scientific Subroutine Library (ESSL)
  * ARM: Arm Performance Libraries
  * HPE: Cray LibSCI
  
* BUT: it is in Fortran, maybe with a C interface
  
### Example BLAS Matrix Vector Multiply

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
  * `0.0`: scaling of `y`
  * `y`: left hand side vector
  * `1`: stride of `y`
  
### Reasons why this is bad

* 11 !! parameters to do `y = A*x`
* hopefully you defined the extern C functor `dgemv` correct to get the fortran link right
  * the actual fortran functions need to take all the scalar parameters as pointers ...
* The type of the scalars is part of the function name (`dgemv` == `double`, `sgemv` == `float`)
   * no mixed support, e.g. `A` and `x` are `float` and `y` is `double`
* No compile time size propagation
  * Machine Learning is all about small operations with compile time known sizes
* Storage Layout of matrix `A` better be Fortran layout
* Scaling parameters change behavior: `0` means to "ignore" the argument (important for NAN propagation for example)

### History of the proposal

* Reviewed by SGs 6/19 in Cologne (Rev 0) and Belfast (Rev 1)
* Reviewed by LEWGi in Cologne/Belfast

### The Philosophy Behind P1673

* Linear Algebra Functions are just another set of algorithms
  * *Propose a free function interface similar to standard algorithms*
* Encapsule data in the minimal fundamental representation of multi dimensional arrays still providing necessary felxibility
  * *Use `mdspan` as parameters*
* Ensure that all functionality of the BLAS standard is covered
  * *Propose equivalent of every function in BLAS*
* Ensure that new areas of linear algebra such as Machine Learning are covered
  * *Turns out mdspan gives that to us naturally*
  
### Example `y=A*x` with P1673

```c++
// Matrix
mdspan<const double,dynamic_extent,dynamic_extent> A(A_ptr,nRows,nCols);
mdspan<const double,dynamic_extent> x(x_ptr,nCols);
mdspan<double,dynamic_extent> y(y_ptr,nRows);

matrix_vector_product(y,A,x);
```

* Only the 3 parameters you would expect
* Function name doesn't encode scalar type

### What about mixed precision?

* Just use the desired scalar types for each argument

```c++
// Matrix
mdspan<const float,dynamic_extent,dynamic_extent> A(A_ptr,nRows,nCols);
mdspan<const float,dynamic_extent> x(x_ptr,nCols);
mdspan<double,dynamic_extent>y(y_ptr,nRows);

// Can infer from return argument y to sum up in double precision
matrix_vector_product(y,A,x);
```

### What about compile time extents

* MDSpan supports compile time extents

```c++
// Matrix
mdspan<const float,8,4> A(A_ptr,nRows,nCols);
mdspan<const float,4> x(x_ptr,nCols);
mdspan<double,8>y(y_ptr,nRows);

// All the dimensions are known at compile time
// Enables full unrolling, vectorization etc. 
matrix_vector_product(y,A,x);
```

### What about scaling parameters e.g. `y = alpha * A * x`

* `basic_mdspan` is flexible enough to represent a scaling factor
  * use a 'scaling accessor' which scales values inline upon access
  
```c++
// Matrix
mdspan<const float,8,4> A(A_ptr,nRows,nCols);
mdspan<const float,4> x(x_ptr,nCols);
mdspan<double,8>y(y_ptr,nRows);

// scaled_view returns a new mdspan with an accessor which 
// multiples elements by 2.0 before returning a value
matrix_vector_product(y,scaled_view(2.0,A),x);
```

* Note: no need to write a different implementation for `matrix_vector_product`

### What about transposing the Matrix (or conjugate transpose etc.)

* Same as with a scaling factor: `basic_mdspan` can accomodate that simply through a change of `layout`

```c++
// Matrix
mdspan<const float,8,4> A(A_ptr,nRows,nCols);
mdspan<const float,4> x(x_ptr,nCols);
mdspan<double,8>y(y_ptr,nRows);

// Transpose_view return a new basic_mdspan with a transposed layout
matrix_vector_product(y,transpose_view(A),x);
```

### What about the stride parameters?

* Covered by `basic_mdspan` and its `layouts` e.g. `layout_stride` 
  * we also propose more specific linear algebra layouts. 
* This covers use cases such as building blocks for tensor algebra, where both dimensions of a matrix need to have a stride
  * The BLAS standard doesn't support that
  
## Content of the proposal

* Glancing over variants here and some minor helper things

### Layouts for `basic_mdspan`

* `layout_blas_packed`
* `layout_blas_general`

### `basic_mdspan` modifier functions

* `scaled_view`
* `tranpose_view`
* `conjugate_view`
* `conjugate_transpose_view`

### Algorithms

* Matrix and Vector algorithms
  * `linalg_swap`
  * `scale`
  * `linalg_copy`
  * `linalg_add`
* Vector operations
  * `givens_rotation_apply`
  * `dot`
  * `vector_norm2`
  * `vector_abs_sum`
  * `idx_abs_max`
* Matrix-Vector algorithms
  * `matrix_vector_product`
  * `symmetric_matrix_vector_product`
  * `hermitian_matrix_vector_product`
  * `triangular_matrix_vector_product`
  * `triangular_matrix_vector_solve`
  * `matrix_rank_1_update`
  * `symmetric_matrix_rank_1_update`
  * `hermitian_matrix_rank_1_update`
  * `symmetric_matrix_rank_2_update`
  * `hermitian_matrix_rank_2_update`
* Matrix-Matrix algorithms
  * `matrix_product`
  * `symmetric_matrix_right_product`
  * `hermitian_matrix_right_product`
  * `triangular_matrix_right_product`
  * `symmetric_matrix_left_product`
  * `hermitian_matrix_left_product`
  * `triangular_matrix_left_product`
  * `symmetric_matrix_rank_k_update`
  * `hermitian_matrix_rank_k_update`
  * `symmetric_matrix_rank_2k_update`
  * `hermitian_matrix_rank_2k_update`  
  * `triangular_matrix_matrix_left_solve`
  * `triangular_matrix_matrix_right_solve`
  

