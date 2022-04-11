
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

template<class MDSpanValueType, class KViewValueType = MDSpanValueType>
void mdspan_to_view_test_impl()
{
  using std::experimental::mdspan;
  using std::experimental::extents;
  using std::experimental::dynamic_extent;

  // rank1, non-const
  {
    std::vector<MDSpanValueType> a(5);
    using mdspan_t = mdspan<MDSpanValueType, extents<dynamic_extent>>;
    mdspan_t mdsp(a.data(), a.size());

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 1);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 5);
  }

  // rank1, const
  {
    std::vector<MDSpanValueType> a(5);
    using mdspan_t = mdspan<const MDSpanValueType, extents<dynamic_extent>>;
    mdspan_t mdsp(a.data(), a.size());

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 1);
    static_assert(std::is_same_v<typename kv_type::value_type, const KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 5);
  }

  // rank2, non-const
  {
    std::vector<MDSpanValueType> a(12);
    using mdspan_t = mdspan<MDSpanValueType, extents<dynamic_extent, dynamic_extent>>;
    mdspan_t mdsp(a.data(), 3, 4);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 3);
    EXPECT_TRUE(kv.extent(1) == 4);
  }

  // rank2, const
  {
    std::vector<MDSpanValueType> a(12);
    using mdspan_t = mdspan<const MDSpanValueType, extents<dynamic_extent, dynamic_extent>>;
    mdspan_t mdsp(a.data(), 3, 4);

    auto kv = KokkosKernelsSTD::Impl::mdspan_to_view(mdsp);
    using kv_type = decltype(kv);
    static_assert(kv_type::rank == 2);
    static_assert(std::is_same_v<typename kv_type::value_type, const KViewValueType>);
    EXPECT_TRUE(kv.extent(0) == 3);
    EXPECT_TRUE(kv.extent(1) == 4);
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
