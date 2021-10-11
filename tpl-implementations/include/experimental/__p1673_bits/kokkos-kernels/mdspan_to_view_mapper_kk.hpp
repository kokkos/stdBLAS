
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MDSPANTOVIEW_MAPPER_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MDSPANTOVIEW_MAPPER_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

// These are only minimal mappers from mdspan to View without safety checks.
// Kokkos is working on using mdspan as the underlying implementation for View
// When that is done the mapping becomes trivial and full featured.
template<class Layout>
struct LayoutMapper;

template<>
struct LayoutMapper<std::experimental::layout_left> {
  using type = Kokkos::LayoutLeft;
};

template<>
struct LayoutMapper<std::experimental::layout_right> {
  using type = Kokkos::LayoutRight;
};

template<class MDSpan>
std::enable_if_t<MDSpan::rank()==1, Kokkos::View<typename MDSpan::element_type*>>
mdspan_to_view(const MDSpan& a) {
    return Kokkos::View<typename MDSpan::element_type*>(a.data(),a.extent(0));
}

template<class MDSpan>
std::enable_if_t<MDSpan::rank()==2, Kokkos::View<typename MDSpan::element_type**,
                                                 typename LayoutMapper<typename MDSpan::layout_type>::type>>
mdspan_to_view(const MDSpan& a) {
    return Kokkos::View<typename MDSpan::element_type**,
                        typename LayoutMapper<typename MDSpan::layout_type>::type>
            (a.data(),a.extent(0),a.extent(1));
}
} // namespace Impl
} // namespac KokkosKernelsSTD
#endif
