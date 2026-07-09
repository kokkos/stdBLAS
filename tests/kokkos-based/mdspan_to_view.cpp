
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

template<class MDSpanType, class KViewType>
void expect_shallow_copy(MDSpanType mdsp, KViewType kv)
{
  EXPECT_EQ( (void *) mdsp.data_handle(), (void *) kv.data() );
}

template<class MDSpanValueType, class KViewValueType = MDSpanValueType>
void mdspan_to_view_test_impl()
{
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::dims;

  // rank1, non-const
  {
    std::vector<MDSpanValueType> a(5);
    using mdspan_t = mdspan<MDSpanValueType, dims<1>>;
    mdspan_t mdsp(a.data(), a.size());

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 1);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 5);
    expect_shallow_copy(mdsp, kv);
  }

  // rank1, const
  {
    std::vector<MDSpanValueType> a(5);
    using mdspan_t = mdspan<const MDSpanValueType, dims<1>>;
    mdspan_t mdsp(a.data(), a.size());

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 1);
    static_assert(std::is_same_v<typename kv_type::value_type, const KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 5);
    expect_shallow_copy(mdsp, kv);
  }

  // rank2, non-const
  {
    std::vector<MDSpanValueType> a(12);
    using mdspan_t = mdspan<MDSpanValueType, dims<2>>;
    mdspan_t mdsp(a.data(), 3, 4);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 3);
    EXPECT_TRUE(kv.extent(1) == 4);
    expect_shallow_copy(mdsp, kv);
  }

  // rank2, const
  {
    std::vector<MDSpanValueType> a(12);
    using mdspan_t = mdspan<const MDSpanValueType, dims<2>>;
    mdspan_t mdsp(a.data(), 3, 4);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, const KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 3);
    EXPECT_TRUE(kv.extent(1) == 4);
    expect_shallow_copy(mdsp, kv);
  }
}

TEST(mdspan_to_view, for_float){
  mdspan_to_view_test_impl<float>();
}

TEST(mdspan_to_view, for_double){
  mdspan_to_view_test_impl<double>();
}

TEST(mdspan_to_view, for_complex_double){
  using value_type = std::complex<double>;
  using kc_t       = Kokkos::complex<double>;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    mdspan_to_view_test_impl<value_type, kc_t>();
  }
}


template<class MDSpanValueType, class KViewValueType = MDSpanValueType>
void transposed_mdspan_to_view_test_impl()
{
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::dims;

  using lr_t = Kokkos::layout_right;
  using ll_t = Kokkos::layout_left;

  std::vector<MDSpanValueType> a(12);
  std::iota(a.begin(), a.end(), 0);

  {
    // mdspan is layout right
    using mdspan_t = mdspan<MDSpanValueType, dims<2>, lr_t>;
    mdspan_t mdsp(a.data(), 3, 4);
    auto mdsp_T = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::linalg::transposed(mdsp);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp_T);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);

    // the conversion from transposed() to view basically discards the transposition
    // so behaves as if we were tranposing the nested mspan directly
    static_assert(std::is_same_v<typename kv_type::array_layout, Kokkos::LayoutLeft>);
    EXPECT_TRUE(kv.extent(0) == 4);
    EXPECT_TRUE(kv.extent(1) == 3);
    expect_shallow_copy(mdsp, kv);

    int count=0;
    // for layoutright we need to loop over each row
    for (std::size_t i=0; i<kv.extent(0); ++i){
      for (std::size_t j=0; j<kv.extent(1); ++j){
	EXPECT_TRUE( mdsp(i,j) == a[count++] );
	EXPECT_TRUE( mdsp(i,j) == kv(i,j) );
      }
    }
  }

  {
    // mdspan is layout left
    using mdspan_t = mdspan<MDSpanValueType, dims<2>, ll_t>;
    mdspan_t mdsp(a.data(), 3, 4);
    auto mdsp_T = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::linalg::transposed(mdsp);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp_T);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);

    // the conversion from transposed() to view basically discards the transposition
    // so behaves as if we were tranposing the nested mspan directly
    static_assert(std::is_same_v<typename kv_type::array_layout, Kokkos::LayoutRight>);
    EXPECT_TRUE(kv.extent(0) == 4);
    EXPECT_TRUE(kv.extent(1) == 3);
    expect_shallow_copy(mdsp, kv);

    int count=0;
    // for layoutleft we need to loop over each columns
    for (std::size_t j=0; j<kv.extent(1); ++j){
      for (std::size_t i=0; i<kv.extent(0); ++i){
	EXPECT_TRUE( mdsp(i,j) == a[count++] );
	EXPECT_TRUE( mdsp(i,j) == kv(i,j) );
      }
    }
  }
}

TEST(mdspan_to_view, float_transposed){
  transposed_mdspan_to_view_test_impl<float>();
}

TEST(mdspan_to_view, double_transposed){
  transposed_mdspan_to_view_test_impl<double>();
}

TEST(mdspan_to_view, complex_double_transposed){
  using value_type = std::complex<double>;
  using kc_t       = Kokkos::complex<double>;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    transposed_mdspan_to_view_test_impl<value_type, kc_t>();
  }
}
