
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MDSPANTOVIEW_MAPPER_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_MDSPANTOVIEW_MAPPER_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

template<class T> struct is_complex : std::false_type{};
template<> struct is_complex<std::complex<float>> : std::true_type{};
template<> struct is_complex<std::complex<double>> : std::true_type{};
template<> struct is_complex<std::complex<long double>> : std::true_type{};

template<class T> inline constexpr bool is_complex_v = is_complex<T>::value;


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
std::enable_if_t<
  MDSpan::rank()==1
  && !is_complex_v<typename MDSpan::element_type>,
  Kokkos::View<typename MDSpan::element_type*>
  >
mdspan_to_view(const MDSpan& a) {
  return Kokkos::View<typename MDSpan::element_type*>(a.data(), a.extent(0));
}

template<class MDSpan>
std::enable_if_t<
  MDSpan::rank()==1
  && is_complex_v<typename MDSpan::element_type>,
  Kokkos::View< Kokkos::complex<typename MDSpan::element_type::value_type>* >
  >
mdspan_to_view(const MDSpan& a)
{
  using mdspan_v_type = typename MDSpan::element_type::value_type;
  using kc_t = Kokkos::complex<mdspan_v_type>;
  if (alignof(mdspan_v_type) != alignof(kc_t)){
    throw std::runtime_error("KokkosBlas: mdspan_to_view: mismatching alignment of std::complex<> and Kokkos::complex<> which prevents interoperability. By default, Kokkos::complex<T> is aligned as 2*sizeof(T) but std::complex<T> is aligned as sizeof(T). To match the alignement and allow correct conversion, you can rebuild Kokkos undefining KOKKOS_ENABLE_COMPLEX_ALIGN.");
  }

  using view_type = Kokkos::View< kc_t* >;
  return view_type(reinterpret_cast<kc_t*>(a.data()), a.extent(0));
}

template<class MDSpan>
std::enable_if_t<
  MDSpan::rank()==2
  && !is_complex_v<typename MDSpan::element_type>,
  Kokkos::View<
    typename MDSpan::element_type **,
    typename LayoutMapper<typename MDSpan::layout_type>::type
    >
  >
mdspan_to_view(const MDSpan& a)
{
  using view_type = Kokkos::View<
    typename MDSpan::element_type **,
    typename LayoutMapper<typename MDSpan::layout_type>::type
    >;
  return view_type(a.data(), a.extent(0), a.extent(1));
}

template<class MDSpan>
std::enable_if_t<
  MDSpan::rank()==2
  && is_complex_v<typename MDSpan::element_type>,
  Kokkos::View<
    Kokkos::complex< typename MDSpan::element_type::value_type> **,
    typename LayoutMapper<typename MDSpan::layout_type>::type
    >
  >
mdspan_to_view(const MDSpan& a)
{
  using mdspan_v_type = typename MDSpan::element_type::value_type;
  using kc_t = Kokkos::complex<mdspan_v_type>;
  if (alignof(mdspan_v_type) != alignof(kc_t)){
    throw std::runtime_error("KokkosBlas: mdspan_to_view: mismatching alignment of std::complex<> and Kokkos::complex<> which prevents interoperability. By default, Kokkos::complex<T> is aligned as 2*sizeof(T) but std::complex<T> is aligned as sizeof(T). To match the alignement and allow correct conversion, you can rebuild Kokkos undefining KOKKOS_ENABLE_COMPLEX_ALIGN.");
  }

  using view_type = Kokkos::View<kc_t**, typename LayoutMapper<typename MDSpan::layout_type>::type>;
  return view_type(reinterpret_cast<kc_t*>(a.data()), a.extent(0), a.extent(1));
}

} // namespace Impl
} // namespac KokkosKernelsSTD
#endif
