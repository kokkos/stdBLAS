 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace{

////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////

// create rank-1 mdspan (vector)
template <typename value_type,
          typename mdspan_t = typename _blas2_signed_fixture<value_type>::mdspan_r1_t>
inline mdspan_t make_mdspan(value_type *data, std::size_t ext) {
  return mdspan_t(data, ext);
}

template <typename value_type,
          typename mdspan_t = typename _blas2_signed_fixture<const value_type>::mdspan_r1_t>
inline mdspan_t make_mdspan(const value_type *data, std::size_t ext) {
  return mdspan_t(data, ext);
}

template <typename value_type>
inline auto make_mdspan(std::vector<value_type> &v) {
  return make_mdspan(v.data(), v.size());
}

template <typename value_type>
inline auto make_mdspan(const std::vector<value_type> &v) {
  return make_mdspan(v.data(), v.size());
}

// create rank-2 mdspan (matrix)
template <typename value_type,
          typename mdspan_t = typename _blas2_signed_fixture<value_type>::mdspan_r2_t>
inline mdspan_t make_mdspan(value_type *data, std::size_t ext0, std::size_t ext1) {
  return mdspan_t(data, ext0, ext1);
}

template <typename ElementType1,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
inline bool is_same_vector(
    const mdspan<ElementType1, extents<dynamic_extent>, LayoutPolicy1, AccessorPolicy1> &v1,
    const mdspan<ElementType2, extents<dynamic_extent>, LayoutPolicy2, AccessorPolicy2> &v2)
{
  const auto size = v1.extent(0);
  if (size != v2.extent(0))
    return false;
  const auto v1_view = KokkosKernelsSTD::Impl::mdspan_to_view(v1);
  const auto v2_view = KokkosKernelsSTD::Impl::mdspan_to_view(v2);
  int diff = false;
  Kokkos::parallel_reduce(size,
    KOKKOS_LAMBDA(const std::size_t i, decltype(diff) &d){
        d = d || !(v1_view(i) == v2_view(i));
	    }, diff);
  return !diff;
}

template <typename ElementType1,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
inline bool is_same_vector(
    const mdspan<ElementType1, extents<dynamic_extent>, LayoutPolicy, AccessorPolicy> &v1,
    const std::vector<ElementType2> &v2)
{
  return is_same_vector(v1, make_mdspan(v2));
}

template <typename ElementType1,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
inline bool is_same_vector(
    const std::vector<ElementType1> &v1,
    const mdspan<ElementType2, extents<dynamic_extent>, LayoutPolicy, AccessorPolicy> &v2)
{
  return is_same_vector(v2, v1);
}

template <typename ElementType>
inline bool is_same_vector(
    const std::vector<ElementType> &v1,
    const std::vector<ElementType> &v2)
{
  return is_same_vector(make_mdspan(v1), make_mdspan(v2));
}

// real diff: d = |v1 - v2|
template <typename T, typename enabled=void>
class value_diff {
public:
  value_diff(const T &val1, const T &val2): _v(fabs(val1 - val2)) {}
  operator T() const { return _v; }
protected:
  value_diff() = default;
  T _v;
};

// real diff: d = max(|R(v1) - R(v2)|, |I(v1) - I(v2)|)
// Note: returned value is of underlying real type
template <typename T>
class value_diff<std::complex<T>>: public value_diff<T> {
  using base = value_diff<T>;
public:
  value_diff(const std::complex<T> &val1, const std::complex<T> &val2) {
    const T dreal = base(val1.real(), val2.real());
    const T dimag = base(val1.imag(), val2.imag());
    base::_v = std::max(dreal, dimag);
  }
};

template <typename T>
class value_diff<Kokkos::complex<T>>: public value_diff<T> {
  using base = value_diff<T>;
public:
  KOKKOS_INLINE_FUNCTION
  value_diff(const Kokkos::complex<T> &val1, const Kokkos::complex<T> &val2) {
    const T dreal = base(val1.real(), val2.real());
    const T dimag = base(val1.imag(), val2.imag());
    base::_v = dreal > dimag ? dreal : dimag; // can't use std::max on GPU
  }
};

template <typename ElementType,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename LayoutPolicy2,
          typename AccessorPolicy2,
          typename ToleranceType>
inline bool is_same_matrix(
    const mdspan<ElementType, extents<dynamic_extent, dynamic_extent>, LayoutPolicy1, AccessorPolicy1> &A,
    const mdspan<ElementType, extents<dynamic_extent, dynamic_extent>, LayoutPolicy2, AccessorPolicy2> &B,
    ToleranceType tolerance)
{
  const auto ext0 = A.extent(0);
  const auto ext1 = A.extent(1);
  if (B.extent(0) != ext0 or B.extent(1) != ext1)
    return false;
  const auto A_view = KokkosKernelsSTD::Impl::mdspan_to_view(A);
  const auto B_view = KokkosKernelsSTD::Impl::mdspan_to_view(B);
  int diff = false;
  Kokkos::parallel_reduce(ext0,
    KOKKOS_LAMBDA(std::size_t i, decltype(diff) &d) {
        for (decltype(i) j = 0; j < ext1; ++j) {
          d = d || (value_diff(A_view(i, j), B_view(i, j)) > tolerance);
        }
	    }, diff);
  return !diff;
}

namespace Impl {

template <typename T, typename enabled=void> struct _tolerance_out { using type = T; };
template <typename T> struct _tolerance_out<std::complex<T>> { using type = T; };

}

template <typename T>
Impl::_tolerance_out<T>::type tolerance(double double_tol, float float_tol);

template <> double tolerance<double>(double double_tol, float float_tol) { return double_tol; }
template <> float  tolerance<float>( double double_tol, float float_tol) { return float_tol; }
template <> double tolerance<std::complex<double>>(double double_tol, float float_tol) { return double_tol; }
template <> float  tolerance<std::complex<float>>( double double_tol, float float_tol) { return float_tol; }

template <typename value_type, typename enabled = void>
struct check_types: public std::true_type {};

// checks if std::complex<T> and Kokkos::complex<T> are aligned
// (they can get misalligned when Kokkos is build with Kokkos_ENABLE_COMPLEX_ALIGN=ON)
template <typename T>
struct check_types<std::complex<T>> {
  static constexpr bool value = alignof(std::complex<T>) == alignof(Kokkos::complex<T>);
};

template <typename value_type>
inline constexpr auto check_types_v = check_types<value_type>::value;

template <typename value_type, typename cb_type>
void run_checked_tests(const std::string_view test_prefix, const std::string_view method_name,
                       const std::string_view test_postfix, const std::string_view type_spec,
                       const cb_type cb) {
  if constexpr (check_types_v<value_type>) {
    cb();
  } else {
    std::cout << "***\n"
              << "***  Warning: " << test_prefix << method_name << test_postfix << " skipped for "
              << type_spec << " (type check failed)\n"
              << "***" << std::endl;
    /* avoid dispatcher check failure if all cases are skipped this way */
    KokkosKernelsSTD::Impl::signal_kokkos_impl_called(method_name);
  }
}

#define FOR_ALL_BLAS2_TYPES(TEST_DEF) \
  TEST_DEF(double) \
  TEST_DEF(float) \
  TEST_DEF(complex_double)


////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////

template<class x_t, class A_t, class Triangle>
void hermitian_matrix_rank_1_update_gold_solution(const x_t &x, A_t &A, Triangle /* t */)
{
  using size_type = std::experimental::extents<>::size_type;
  constexpr auto conj = std::experimental::linalg::impl::conj_if_needed;
  constexpr bool low = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;
  for (size_type j = 0; j < A.extent(1); ++j) {
    const size_type i1 = low ? A.extent(0) : j + 1;
    for (size_type i = low ? j : 0; i < i1; ++i) {
      A(i,j) += x(i) * conj(x(j));
    }
  }
}

template<class x_t, class A_t, typename gold_t, typename action_t>
void test_kokkos_matrix_update(const x_t &x, A_t &A, gold_t get_gold, action_t action)
{
  using value_type = typename x_t::value_type;

  // backup x to verify it is not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);

  // compute gold
  auto A_copy = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto A_gold = make_mdspan(A_copy.data(), A.extent(0), A.extent(1));
  get_gold(A_gold);

  // run tested routine
  action();

  // compare results with gold
  EXPECT_TRUE(is_same_matrix(A_gold, A, tolerance<value_type>(1e-9, 1e-2f)));

  // x should not change after kernel
  EXPECT_TRUE(is_same_vector(x, x_preKernel));
}

template<class x_t, class A_t, class Triangle, class Scalar = typename x_t::element_type>
void test_kokkos_hermitian_matrix_rank1_update_impl(const x_t &x, A_t &A, Triangle t)
{
  const auto get_gold = [&](auto A_gold) {
      hermitian_matrix_rank_1_update_gold_solution(x, A_gold, t);
    };
  const auto compute = [&]() {
      std::experimental::linalg::hermitian_matrix_rank_1_update(
        KokkosKernelsSTD::kokkos_exec<>(), x, A, t);
    };
  test_kokkos_matrix_update(x, A, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_hermitian_matrix_rank1_update) {                               \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "hermitian_matrix_rank1_update", "",   \
                           #blas_val_type, [&]() {                           \
                                                                             \
    test_kokkos_hermitian_matrix_rank1_update_impl(x_e0, A_sym_e0,           \
                            std::experimental::linalg::lower_triangle);      \
    test_kokkos_hermitian_matrix_rank1_update_impl(x_e0, A_sym_e0,           \
                            std::experimental::linalg::upper_triangle);      \
                                                                             \
  });                                                                        \
}

FOR_ALL_BLAS2_TYPES(DEFINE_TESTS);
