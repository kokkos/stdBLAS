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

#ifndef LINALG_TESTS_KOKKOS_HELPERS_HPP_
#define LINALG_TESTS_KOKKOS_HELPERS_HPP_

#include <experimental/linalg>
#include <experimental/mdspan>
#include <random>

namespace kokkostesting{

template<class T>
auto create_stdvector_and_copy(T sourceView)
{
  static_assert (sourceView.rank() == 1);

  using value_type = typename T::value_type;
  using res_t = std::vector<value_type>;

  res_t result(sourceView.extent(0));
  for (std::size_t i=0; i<sourceView.extent(0); ++i){
    result[i] = sourceView(i);
  }

  return result;
}

template<class T>
auto create_stdvector_and_copy_rowwise(T sourceView)
{
  static_assert (sourceView.rank() == 2);

  using value_type = typename T::value_type;
  using res_t = std::vector<value_type>;

  res_t result(sourceView.extent(0)*sourceView.extent(1));
  std::size_t k=0;
  for (std::size_t i=0; i<sourceView.extent(0); ++i){
    for (std::size_t j=0; j<sourceView.extent(1); ++j){
      result[k++] = sourceView(i,j);
    }
  }

  return result;
}

// create rank-1 mdspan (vector)
template <typename ValueType,
          typename mdspan_t = typename _blas2_signed_fixture<ValueType>::mdspan_r1_t>
mdspan_t make_mdspan(ValueType *data, std::size_t ext) {
  return mdspan_t(data, ext);
}

template <typename ValueType>
auto make_mdspan(std::vector<ValueType> &v) {
  return make_mdspan(v.data(), v.size());
}

template <typename ValueType>
auto make_mdspan(const std::vector<ValueType> &v) {
  return make_mdspan(v.data(), v.size());
}

// create rank-2 mdspan (matrix)
template <typename ValueType,
          typename mdspan_t = typename _blas2_signed_fixture<ValueType>::mdspan_r2_t>
mdspan_t make_mdspan(ValueType *data, std::size_t ext0, std::size_t ext1) {
  return mdspan_t(data, ext0, ext1);
}

namespace Impl {

template <typename ElementType,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy>
auto abs_max(mdspan<ElementType, extents<Extent>, LayoutPolicy, AccessorPolicy> v)
{
  const auto size = v.extent(0);
  if (size == 0) {
    throw std::runtime_error("abs_max() requires non-empty input");
  }
  const auto i = std::experimental::linalg::idx_abs_max(v);
  if (i >= size) { // shouldn't happen: empty case is handled above
    throw std::runtime_error("Fatal: idx_abs_max() failed");
  }
  return std::abs(v[i]);
}

template <typename ElementType,
          std::size_t Extent0,
          std::size_t Extent1,
          typename LayoutPolicy,
          typename AccessorPolicy>
auto abs_max(mdspan<ElementType, extents<Extent0, Extent1>, LayoutPolicy, AccessorPolicy> A)
{
  const auto ext0 = A.extent(0);
  const auto ext1 = A.extent(1);
  if (ext0 == 0 or ext1 == 0) {
    throw std::runtime_error("abs_max() requires non-empty input");
  }
  const auto A_view = KokkosKernelsSTD::Impl::mdspan_to_view(A);
  using RetType = decltype(Kokkos::abs(A_view(0, 0)));
  RetType result;
  const auto red = Kokkos::Max<RetType>(result);
  Kokkos::parallel_reduce(ext0,
    KOKKOS_LAMBDA(std::size_t i, RetType &max_val) {
        for (decltype(i) j = 0; j < ext1; ++j) {
          red.join(max_val, Kokkos::abs(A_view(i, j)));
        }
	    }, red);
  return result;
}

template <typename RealType>
RealType abs2rel_diff(RealType abs_diff, RealType norm1, RealType norm2)
{
  constexpr auto zero = static_cast<RealType>(0);
  if (norm1 != zero and norm2 != zero) {
    return abs_diff / std::min(norm1, norm2); // pick larger relative error
  } else if (norm1 == zero and norm2 == zero) {
    return zero; // no difference
  }
  // Can't get good relative diff with zero -
  // so return absolute diff out of better ideas...
  return abs_diff;
}

}

// no-tolerance (exact) comparison
template <typename ElementType1,
          std::size_t Extent1,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          std::size_t Extent2,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
bool is_same_vector(
    mdspan<ElementType1, extents<Extent1>, LayoutPolicy1, AccessorPolicy1> v1,
    mdspan<ElementType2, extents<Extent2>, LayoutPolicy2, AccessorPolicy2> v2)
{
  const auto size = v1.extent(0);
  if (size != v2.extent(0))
    return false;
  const auto v1_view = KokkosKernelsSTD::Impl::mdspan_to_view(v1);
  const auto v2_view = KokkosKernelsSTD::Impl::mdspan_to_view(v2);
  // Note: reducing to `int` because Kokkos can complain on `bool` not being
  //       aligned with int32 and deny it for parallel_reduce()
  using diff_type = int;
  diff_type is_different = false;
  const auto red = Kokkos::LOr<diff_type>(is_different);
  Kokkos::parallel_reduce(size,
    KOKKOS_LAMBDA(const std::size_t i, diff_type &diff){
        red.join(diff, v1_view[i] != v2_view[i]);
	    }, red);
  return !is_different;
}

template <typename ElementType1,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
bool is_same_vector(
    mdspan<ElementType1, extents<Extent>, LayoutPolicy, AccessorPolicy> v1,
    const std::vector<ElementType2> &v2)
{
  return is_same_vector(v1, make_mdspan(v2));
}

template <typename ElementType1,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
bool is_same_vector(
    const std::vector<ElementType1> &v1,
    mdspan<ElementType2, extents<Extent>, LayoutPolicy, AccessorPolicy> v2)
{
  return is_same_vector(v2, v1);
}

template <typename ElementType>
bool is_same_vector(
    const std::vector<ElementType> &v1,
    const std::vector<ElementType> &v2)
{
  return is_same_vector(make_mdspan(v1), make_mdspan(v2));
}

template <typename ElementType1,
          std::size_t Extent1,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          std::size_t Extent2,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
auto vector_abs_diff(
    mdspan<ElementType1, extents<Extent1>, LayoutPolicy1, AccessorPolicy1> v1,
    mdspan<ElementType2, extents<Extent2>, LayoutPolicy2, AccessorPolicy2> v2)
{
  using RetType = decltype(std::abs(v1[0] - v2[0])); // will be same for views
  const auto size = v1.extent(0);
  if (size != v2.extent(0)) {
    throw std::runtime_error("Compared vectors have different sizes");
  } else if (size == 0) {
    return static_cast<RetType>(0); // no difference
  }
  const auto v1_view = KokkosKernelsSTD::Impl::mdspan_to_view(v1);
  const auto v2_view = KokkosKernelsSTD::Impl::mdspan_to_view(v2);
  RetType difference;
  const auto red = Kokkos::Max<RetType>(difference);
  Kokkos::parallel_reduce(size,
    KOKKOS_LAMBDA(const std::size_t i, RetType &diff){
        const auto val1 = v1_view[i];
        const auto val2 = v2_view[i];
        red.join(diff, Kokkos::abs(val1 - val2));
	    }, red);
  return difference;
}

template <typename ElementType1,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
auto vector_abs_diff(
    mdspan<ElementType1, extents<Extent>, LayoutPolicy, AccessorPolicy> v1,
    const std::vector<ElementType2> &v2)
{
  return vector_abs_diff(v1, make_mdspan(v2));
}

template <typename ElementType1,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
auto vector_abs_diff(
    const std::vector<ElementType1> &v1,
    mdspan<ElementType2, extents<Extent>, LayoutPolicy, AccessorPolicy> v2)
{
  return vector_abs_diff(v2, v1);
}

template <typename ElementType1, typename ElementType2>
auto vector_abs_diff(
    const std::vector<ElementType1> &v1,
    const std::vector<ElementType2> &v2)
{
  return vector_abs_diff(make_mdspan(v1), make_mdspan(v2));
}

template <typename ElementType1,
          std::size_t Extent1,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          std::size_t Extent2,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
auto vector_rel_diff(
    mdspan<ElementType1, extents<Extent1>, LayoutPolicy1, AccessorPolicy1> v1,
    mdspan<ElementType2, extents<Extent2>, LayoutPolicy2, AccessorPolicy2> v2)
{
  using RetType = decltype(std::abs(v1[0] - v2[0]));
  const auto size = v1.extent(0);
  if (size != v2.extent(0)) {
    throw std::runtime_error("Compared vectors have different sizes");
  } else if (size == 0) {
    return static_cast<RetType>(0); // both empty -> no difference
  }
  const auto abs_diff = vector_abs_diff(v1, v2);
  const auto max1 = Impl::abs_max(v1);
  const auto max2 = Impl::abs_max(v2);
  return Impl::abs2rel_diff(abs_diff, max1, max2);
}

template <typename ElementType1,
          std::size_t Extent1,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
auto vector_rel_diff(
    mdspan<ElementType1, extents<Extent1>, LayoutPolicy, AccessorPolicy> v1,
    const std::vector<ElementType2> &v2)
{
  return vector_rel_diff(v1, make_mdspan(v2));
}

template <typename ElementType1,
          std::size_t Extent,
          typename LayoutPolicy,
          typename AccessorPolicy,
          typename ElementType2>
auto vector_rel_diff(
    const std::vector<ElementType1> &v1,
    mdspan<ElementType2, extents<Extent>, LayoutPolicy, AccessorPolicy> v2)
{
  return vector_rel_diff(v2, v1);
}

template <typename ElementType1, typename ElementType2>
auto vector_rel_diff(
    const std::vector<ElementType1> &v1,
    const std::vector<ElementType2> &v2)
{
  return vector_rel_diff(make_mdspan(v1), make_mdspan(v2));
}

// no-tolerance (exact) comparison
template <typename ElementType1,
          std::size_t Extent10,
          std::size_t Extent11,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          std::size_t Extent20,
          std::size_t Extent21,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
bool is_same_matrix(
    mdspan<ElementType1, extents<Extent10, Extent11>, LayoutPolicy1, AccessorPolicy1> A,
    mdspan<ElementType2, extents<Extent20, Extent21>, LayoutPolicy2, AccessorPolicy2> B)
{
  const auto ext0 = A.extent(0);
  const auto ext1 = A.extent(1);
  if (B.extent(0) != ext0 or B.extent(1) != ext1)
    return false;
  const auto A_view = KokkosKernelsSTD::Impl::mdspan_to_view(A);
  const auto B_view = KokkosKernelsSTD::Impl::mdspan_to_view(B);
  // Note: reducing to `int` because Kokkos can complain on `bool` not being
  //       aligned with int32 and deny it for parallel_reduce()
  using diff_type = int;
  diff_type is_different = false;
  Kokkos::parallel_reduce(ext0,
    KOKKOS_LAMBDA(std::size_t i, diff_type &diff) {
        for (decltype(i) j = 0; j < ext1; ++j) {
          const bool d = A_view(i, j) != B_view(i, j);
          diff = diff || d;
        }
	    }, Kokkos::LOr<diff_type>(is_different));
  return !is_different;
}

template <typename ElementType,
          std::size_t Extent0,
          std::size_t Extent1,
          typename LayoutPolicy1,
          typename AccessorPolicy1>
bool is_same_matrix(
    mdspan<ElementType, extents<Extent0, Extent1>, LayoutPolicy1, AccessorPolicy1> A,
    const std::vector<ElementType> &B)
{
  return is_same_matrix(A, make_mdspan(B.data(), A.extent(0), A.extent(1)));
}

template <typename ElementType,
          std::size_t Extent0,
          std::size_t Extent1,
          typename LayoutPolicy1,
          typename AccessorPolicy1>
bool is_same_matrix(const std::vector<ElementType> &A,
    mdspan<ElementType, extents<Extent0, Extent1>, LayoutPolicy1, AccessorPolicy1> B)
{
  return is_same_matrix(make_mdspan(A.data(), B.extent(0), B.extent(1)), B);
}

template <typename ElementType1,
          std::size_t Extent10,
          std::size_t Extent11,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          typename ElementType2,
          std::size_t Extent20,
          std::size_t Extent21,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
auto matrix_abs_diff(
    mdspan<ElementType1, extents<Extent10, Extent11>, LayoutPolicy1, AccessorPolicy1> A,
    mdspan<ElementType2, extents<Extent20, Extent21>, LayoutPolicy2, AccessorPolicy2> B)
{
  using RetType = decltype(std::abs(A(0, 0) - B(0, 0))); // will be same for views
  const auto ext0 = A.extent(0);
  const auto ext1 = A.extent(1);
  if (B.extent(0) != ext0 or B.extent(1) != ext1) {
    throw std::runtime_error("Compared matrices have different sizes");
  } else if (ext0 == 0 or ext1 == 0) {
    return static_cast<RetType>(0); // both empty -> no difference
  }
  const auto A_view = KokkosKernelsSTD::Impl::mdspan_to_view(A);
  const auto B_view = KokkosKernelsSTD::Impl::mdspan_to_view(B);
  RetType difference;
  const auto red = Kokkos::Max<RetType>(difference);
  Kokkos::parallel_reduce(ext0,
    KOKKOS_LAMBDA(const std::size_t i, RetType &diff){
        for (size_t j = 0; j < ext1; ++j) {
          const auto a = A_view(i, j);
          const auto b = B_view(i, j);
          red.join(diff, Kokkos::abs(a - b));
        }
	    }, red);
  return difference;
}

template <typename ElementType,
          std::size_t Extent10,
          std::size_t Extent11,
          typename LayoutPolicy1,
          typename AccessorPolicy1,
          std::size_t Extent20,
          std::size_t Extent21,
          typename LayoutPolicy2,
          typename AccessorPolicy2>
auto matrix_rel_diff(
    mdspan<ElementType, extents<Extent10, Extent11>, LayoutPolicy1, AccessorPolicy1> A,
    mdspan<ElementType, extents<Extent20, Extent21>, LayoutPolicy2, AccessorPolicy2> B)
{
  using RetType = decltype(std::abs(A(0, 0) - B(0, 0)));
  const auto ext0 = A.extent(0);
  const auto ext1 = A.extent(1);
  if (B.extent(0) != ext0 or B.extent(1) != ext1) {
    throw std::runtime_error("Compared matrices have different sizes");
  } else if (ext0 == 0 or ext1 == 0) {
    return static_cast<RetType>(0); // both empty -> no difference
  }
  const auto abs_diff = matrix_abs_diff(A, B);
  const auto max1 = Impl::abs_max(A);
  const auto max2 = Impl::abs_max(B);
  return Impl::abs2rel_diff(abs_diff, max1, max2);
}

namespace Impl { // internal to test helpers

template <typename T, typename Enabled=void> struct _tolerance_out { using type = T; };
template <typename T> struct _tolerance_out<std::complex<T>> { using type = T; };

}

// uses T to select single or double precision value
template <typename T>
Impl::_tolerance_out<T>::type tolerance(double double_tol, float float_tol);

template <> double tolerance<double>(double double_tol, float float_tol) { return double_tol; }
template <> float  tolerance<float>( double double_tol, float float_tol) { return float_tol; }
template <> double tolerance<std::complex<double>>(double double_tol, float float_tol) { return double_tol; }
template <> float  tolerance<std::complex<float>>( double double_tol, float float_tol) { return float_tol; }

// checks if std::complex<T> and Kokkos::complex<T> are aligned
// (they can get misalligned when Kokkos is build with Kokkos_ENABLE_COMPLEX_ALIGN=ON)
template <typename ValueType, typename Enabled = void>
struct check_complex_alignment: public std::true_type {};

template <typename T>
struct check_complex_alignment<std::complex<T>> {
  static constexpr bool value = alignof(std::complex<T>) == alignof(Kokkos::complex<T>);
};

template <typename ValueType>
constexpr auto check_complex_alignment_v = check_complex_alignment<ValueType>::value;

// skips test execution (giving a warning instead) if type checks fail
template <typename ValueType, typename cb_type>
void run_checked_tests(const std::string_view test_prefix, const std::string_view method_name,
                       const std::string_view test_postfix, const std::string_view type_spec,
                       const cb_type cb) {
  if constexpr (check_complex_alignment_v<ValueType>) { // add more checks if needed
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

// drives A = F(A, x, ...) operation test
template<class x_t, class A_t, class AToleranceType, class GoldType, class ActionType>
void test_op_Ax(x_t x, A_t A, AToleranceType A_tol, GoldType get_gold, ActionType action)
{
  // backup x to verify it is not changed after kernel
  auto x_preKernel = create_stdvector_and_copy(x);

  // compute gold
  auto A_copy = create_stdvector_and_copy_rowwise(A);
  auto A_gold = make_mdspan(A_copy.data(), A.extent(0), A.extent(1));
  get_gold(A_gold);

  // run tested routine
  action();

  // compare results with gold
  EXPECT_LE(matrix_rel_diff(A_gold, A), A_tol);

  // x should not change after kernel
  EXPECT_TRUE(is_same_vector(x, x_preKernel));
}

// drives A = F(A, x, y, ...) operation test
template<class x_t, class y_t, class A_t, class AToleranceType, class GoldType, class ActionType>
void test_op_Axy(x_t x, y_t y, A_t A, AToleranceType A_tol, GoldType get_gold, ActionType action)
{
  auto y_preKernel = create_stdvector_and_copy(y);
  test_op_Ax(x, A, A_tol, get_gold, action);
  EXPECT_TRUE(is_same_vector(y, y_preKernel));
}

}
#endif
