
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
  bool diff = false;
  Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(const std::size_t i, bool &diff){
        diff = diff || !(v1(i) == v2(i));
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
  bool diff = false;
  const auto size = ext0 * ext1;
  Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(const std::size_t ij, bool &diff) {
        const auto i = ij / ext1;
        const auto j = ij - i * ext1;
        if (value_diff(A(i, j), B(i, j)) > tolerance)
          diff = true;
	    }, diff);
  return !diff;
}

template <typename T>
struct tolerance {};

template<>
struct tolerance<float> {
  static constexpr float value = 1e-2f;
};

template<>
struct tolerance<double> {
  static constexpr double value = 1e-9;
};

template<>
struct tolerance<std::complex<float>>: public tolerance<float> {};

template<>
struct tolerance<std::complex<double>>: public tolerance<double> {};

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

#define FOR_ALL_BLAS2_TYPES(TEST_DEF) \
  TEST_DEF(double) \
  TEST_DEF(float) \
  TEST_DEF(complex_double)


////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////

template<class x_t, class y_t, class A_t>
void matrix_rank_1_update_gold_solution(const x_t &x, const y_t &y, A_t &A)
{
  using size_type = std::experimental::extents<>::size_type;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      A(i, j) += x(i) * y(j);
    }
  }
}

template<class x_t, class y_t, class A_t>
void kokkos_matrix_rank1_update_impl(const x_t &x, const y_t &y, A_t &A)
{
  using value_type = typename x_t::value_type;

  // backup x and y to verify it is not changed after kernel
  auto x_preKernel = kokkostesting::create_stdvector_and_copy(x);
  auto y_preKernel = kokkostesting::create_stdvector_and_copy(y);

  // compute gold
  auto A_copy = kokkostesting::create_stdvector_and_copy_rowwise(A);
  auto A_gold = make_mdspan(A_copy.data(), A.extent(0), A.extent(1));
  matrix_rank_1_update_gold_solution(x, y, A_gold);

  // run tested routine
  std::experimental::linalg::matrix_rank_1_update(
      KokkosKernelsSTD::kokkos_exec<>(), x, y, A);

  // compare results with gold
  EXPECT_TRUE(is_same_matrix(A_gold, A, tolerance<value_type>::value));

  // x,y should not change after kernel
  EXPECT_TRUE(is_same_vector(x, x_preKernel));
  EXPECT_TRUE(is_same_vector(y, y_preKernel));
}

template <typename value_type, typename cb_type>
void run_checked_tests(const char *type_spec, const cb_type cb) {
  if constexpr (!check_types_v<value_type>) {
    std::cout << "***\n"
              << "***  Warning: kokkos_matrix_rank1_update skipped for " << type_spec << " (type check failed)\n"
              << "***" << std::endl; \
    /* avoid dispatcher check failure if all cases are skipped this way */
    std::cout << "matrix_rank1_update: kokkos impl\n"; \
    return;
  }
  cb();
}

} // anonymous namespace

#define DEFINE_TEST(blas_val_type)                                           \
TEST_F(blas2_signed_##blas_val_type##_fixture, kokkos_matrix_rank1_update) { \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>(#blas_val_type, [&]() {                           \
                                                                             \
    kokkos_matrix_rank1_update_impl(x_e0, x_e1, A_e0e1);                     \
                                                                             \
  });                                                                        \
}

FOR_ALL_BLAS2_TYPES(DEFINE_TEST);
