/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

#ifdef LINALG_ENABLE_BLAS
namespace {

extern "C" void
dgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const double* pALPHA,
        const double* A, const int* pLDA,
        const double* B, const int* pLDB,
        const double* pBETA,
        double* C, const int* pLDC);

extern "C" void
sgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const float* pALPHA,
        const float* A, const int* pLDA,
        const float* B, const int* pLDB,
        const float* pBETA,
        float* C, const int* pLDC);

extern "C" void
cgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const void* pALPHA,
        const void* A, const int* pLDA,
        const void* B, const int* pLDB,
        const void* pBETA,
        void* C, const int* pLDC);

extern "C" void
zgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const void* pALPHA,
        const void* A, const int* pLDA,
        const void* B, const int* pLDB,
        const void* pBETA,
        void* C, const int* pLDC);

template<class Scalar>
struct BlasGemm {
  static constexpr bool supported = false;
  // static void
  // gemm (const char TRANSA[], const char TRANSB[],
  //       const int M, const int N, const int K,
  //       const Scalar ALPHA,
  //       const Scalar* A, const int LDA,
  //       const Scalar* B, const int LDB,
  //       const Scalar BETA,
  //       Scalar* C, const int LDC);
};

template<>
struct BlasGemm<double> {
  static constexpr bool supported = true;
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const double ALPHA,
        const double* A, const int LDA,
        const double* B, const int LDB,
        const double BETA,
        double* C, const int LDC)
  {
    dgemm_ (TRANSA, TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<float> {
  static constexpr bool supported = true;
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const float ALPHA,
        const float* A, const int LDA,
        const float* B, const int LDB,
        const float BETA,
        float* C, const int LDC)
  {
    sgemm_ (TRANSA, TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<std::complex<double>> {
  static constexpr bool supported = true;
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const std::complex<double> ALPHA,
        const std::complex<double>* A, const int LDA,
        const std::complex<double>* B, const int LDB,
        const std::complex<double> BETA,
        std::complex<double>* C, const int LDC)
  {
    zgemm_ (TRANSA, TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<std::complex<float>> {
  static constexpr bool supported = true;
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const std::complex<float> ALPHA,
        const std::complex<float>* A, const int LDA,
        const std::complex<float>* B, const int LDB,
        const std::complex<float> BETA,
        std::complex<float>* C, const int LDC)
  {
    cgemm_ (TRANSA, TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<class in_matrix_t>
constexpr bool valid_input_blas_accessor (in_matrix_t /* A */)
{
  using element_type = typename in_matrix_t::element_type;
  using accessor_type = typename in_matrix_t::accessor_type;

  // The accessor types need not be the same.
  // Input matrices may be scaled or transposed.
  return std::is_same_v<accessor_type,
                        accessor_basic<element_type>> ||
    std::is_same_v<accessor_type,
                   accessor_scaled<accessor_basic<element_type>,
                                   element_type>> ||
    std::is_same_v<accessor_type,
                   accessor_conjugate<accessor_basic<element_type>,
                                      element_type>>;
}

template<class inout_matrix_t>
constexpr bool valid_output_blas_accessor (inout_matrix_t /* C */)
{
  using element_type = typename inout_matrix_t::element_type;
  using accessor_type = typename inout_matrix_t::accessor_type;

  return std::is_same_v<accessor_type,
                        accessor_basic<element_type>>;
}

template<class in_matrix_t>
constexpr bool valid_input_blas_layout (in_matrix_t /* A */)
{
  // Either input matrix may have a transposed layout, but the
  // underlying layout of all matrices must be layout_left (or
  // layout_blas_general<column_major_t>, once we finish implementing
  // that).  layout_right and layout_blas_general<row_major_t> can
  // work if you pretend they are transposes, but we don't try that
  // optimization here.  Another option is a CBLAS implementation,
  // which admits row-major as well as column-major matrices.
  return std::is_same_v<typename in_matrix_t::layout_type,
                   layout_left> ||
    std::is_same_v<typename in_matrix_t::layout_type,
                   layout_transpose<layout_left>>;
    /* || std::is_same_v<typename in_matrix_t::layout_type,
                   layout_blas_general<column_major_t>> ||
      std::is_same_v<typename in_matrix_t::layout_type,
      layout_transpose<layout_blas_general<column_major_t>>>; */
}

template<class inout_matrix_t>
constexpr bool valid_output_blas_layout (inout_matrix_t /* A */)
{
  using layout_type = typename inout_matrix_t::layout_type;
  return std::is_same_v<layout_type, layout_left>;
    /* || std::is_same_v<layout_type, layout_blas_general<column_major_t>>; */
}


template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
constexpr bool
valid_blas_element_types(
  in_matrix_1_t /* A */,
  in_matrix_2_t /* B */,
  out_matrix_t /* C */)
{
  using element_type = typename out_matrix_t::element_type;
  constexpr bool elt_types_same =
    std::is_same_v<element_type,
                   typename in_matrix_1_t::element_type> &&
    std::is_same_v<element_type,
                   typename in_matrix_2_t::element_type>;
  constexpr bool elt_type_ok =
    std::is_same_v<element_type, double> ||
    std::is_same_v<element_type, float> ||
    std::is_same_v<element_type, std::complex<double>> ||
    std::is_same_v<element_type, std::complex<float>>;
  return elt_types_same && elt_type_ok;
}

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
constexpr bool
matrix_product_dispatch_to_blas(
  in_matrix_1_t A,
  in_matrix_2_t B,
  out_matrix_t C)
{
  // The accessor types need not be the same.
  // Input matrices may be scaled or transposed.
  constexpr bool in1_acc_type_ok = valid_input_blas_accessor(A);
  constexpr bool in2_acc_type_ok = valid_input_blas_accessor(B);
  constexpr bool out_acc_type_ok = valid_output_blas_accessor(C);

  constexpr bool in1_layout_ok = valid_input_blas_layout(A);
  constexpr bool in2_layout_ok = valid_input_blas_layout(B);
  constexpr bool out_layout_ok = valid_output_blas_layout(C);

  return C.rank_dynamic() == 2 &&
    valid_blas_element_types(A, B, C) &&
    in1_acc_type_ok && in2_acc_type_ok && out_acc_type_ok &&
    in1_layout_ok && in2_layout_ok && out_layout_ok;
}

}

#endif // LINALG_ENABLE_BLAS

// Overwriting general matrix-matrix product

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
void matrix_product(in_matrix_1_t A,
                    in_matrix_2_t B,
                    out_matrix_t C)
{
#ifdef LINALG_ENABLE_BLAS
  if constexpr (matrix_product_dispatch_to_blas(A, B, C)) {
    // FIXME I'm assuming here that all element types are the same.
    // Classic BLAS assumes that, but we could be using
    // mixed-precision (X) BLAS.
    using element_type = typename out_matrix_t::element_type;

    using A_layout_type = typename in_matrix_1_t::layout_type;
    constexpr bool A_trans =
      std::is_same_v<A_layout_type, layout_transpose<layout_left>>;
      /* || std::is_same_v<A_layout_type, layout_transpose<layout_blas_general<column_major_t>>> */
    using A_acc_type = typename in_matrix_1_t::accessor_type;
    constexpr bool A_conj =
      std::is_same_v<A_acc_type,
                     accessor_conjugate<accessor_basic<element_type>,
                                        element_type>>;
    const char TRANSA = A_trans ? (A_conj ? 'C' : 'T') : 'N';

    using B_layout_type = typename in_matrix_2_t::layout_type;
    constexpr bool B_trans =
      std::is_same_v<B_layout_type, layout_transpose<layout_left>>;
      /* || std::is_same_v<B_layout_type, layout_transpose<layout_blas_general<column_major_t>>> */
    using B_acc_type = typename in_matrix_2_t::accessor_type;
    constexpr bool B_conj =
      std::is_same_v<B_acc_type,
                     accessor_conjugate<accessor_basic<element_type>,
                                        element_type>>;
    const char TRANSB = B_trans ? (B_conj ? 'C' : 'T') : 'N';

    const int M = C.extent(0);
    const int N = C.extent(1);
    const int K = A.extent(1);

    // TODO extract scalars from accessor_scaled.
    const element_type alpha (1.0);
    const element_type beta (0.0);

    static_assert(A.is_strided() && B.is_strided() && C.is_strided());
    const int LDA = A.stride(1) == 0 ? 1 : int(A.stride(1));
    const int LDB = B.stride(1) == 0 ? 1 : int(B.stride(1));
    const int LDC = C.stride(1) == 0 ? 1 : int(C.stride(1));
    BlasGemm<element_type>::gemm(TRANSA, TRANSB, M, N, K,
                                 alpha, A.data(), LDA,
                                 B.data(), LDB,
                                 beta, C.data(), LDC);
  }
  else
#endif // LINALG_ENABLE_BLAS
  {
    for(ptrdiff_t i = 0; i < C.extent(0); ++i) {
      for(ptrdiff_t j = 0; j < C.extent(1); ++j) {
        C(i,j) = 0.0;
        for(ptrdiff_t k = 0; k < A.extent(1); ++k) {
          C(i,j) += A(i,k) * B(k,j);
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
void matrix_product(ExecutionPolicy&& /* exec */,
                    in_matrix_1_t A,
                    in_matrix_2_t B,
                    out_matrix_t C)
{
  matrix_product(A, B, C);
}

// Updating general matrix-matrix product

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void matrix_product(in_matrix_1_t A,
                    in_matrix_2_t B,
                    in_matrix_3_t E,
                    out_matrix_t C)
{
  for(ptrdiff_t i = 0; i < C.extent(0); ++i) {
    for(ptrdiff_t j = 0; j < C.extent(1); ++j) {
      C(i,j) = E(i,j);
      for(ptrdiff_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void matrix_product(ExecutionPolicy&& /* exec */,
                    in_matrix_1_t A,
                    in_matrix_2_t B,
                    in_matrix_3_t E,
                    out_matrix_t C)
{
  matrix_product(A, B, E, C);
}

// Overwriting symmetric matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void symmetric_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void symmetric_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  symmetric_matrix_product(A, t, s, B, C);
}

// Updating symmetric matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void symmetric_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void symmetric_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  symmetric_matrix_product(A, t, s, B, E, C);
}

// Overwriting Hermitian matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void hermitian_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void hermitian_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  hermitian_matrix_product(A, t, s, B, C);
}

// Updating Hermitian matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void hermitian_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = E(i,j);
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void hermitian_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  hermitian_matrix_product(A, t, s, B, E, C);
}

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
