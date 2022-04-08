
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

template<class T>
constexpr void static_assert_complex_alignment(const T* /* p */)
{
  using kc_t = Kokkos::complex<typename T::value_type>;
  static_assert(alignof(T) == alignof(kc_t),
    "KokkosBlas: mdspan_to_view: mismatching alignment of std::complex<> and Kokkos::complex<> which prevents interoperability. By default, Kokkos::complex<T> is aligned as 2*sizeof(T) but std::complex<T> is aligned as sizeof(T). To match the alignment and allow correct conversion, you must rebuild Kokkos with KOKKOS_ENABLE_COMPLEX_ALIGN undefined.");
}

//
// to_kokkos_pointer
//
template<class ValueType>
ValueType* to_kokkos_pointer(ValueType* p) {
  return p;
}

template<class RealType>
auto to_kokkos_pointer(std::complex<RealType>* p) {
  static_assert_complex_alignment(p);
  return reinterpret_cast<Kokkos::complex<RealType>*>(p);
}

template<class RealType>
auto to_kokkos_pointer(const std::complex<RealType>* p) {
  static_assert_complex_alignment(p);
  return reinterpret_cast<const Kokkos::complex<RealType>*>(p);
}

//
// mdspan_to_view
//
template<
  class ElementType,
  std::experimental::extents<>::size_type ext,
  class Layout,
  class Accessor>
auto mdspan_to_view(std::experimental::mdspan<
          ElementType,
          std::experimental::extents<ext>,
          Layout,
          Accessor
        > a)
{
  auto kokkos_p = to_kokkos_pointer(a.data());
  return Kokkos::View<decltype(kokkos_p)>(kokkos_p, a.extent(0));
}

template<
  class ElementType,
  std::experimental::extents<>::size_type ext0,
  std::experimental::extents<>::size_type ext1,
  class Layout,
  class Accessor>
auto mdspan_to_view(std::experimental::mdspan<
          ElementType,
          std::experimental::extents<ext0, ext1>,
          Layout,
          Accessor
        > a)
{
  using mdspan_type = std::experimental::mdspan<
    ElementType, std::experimental::extents<ext0, ext1>, Layout, Accessor
    >;

  auto kokkos_p = to_kokkos_pointer(a.data());
  using view_type = Kokkos::View<
    decltype(kokkos_p)*,
    typename LayoutMapper<typename mdspan_type::layout_type>::type
    >;
  return view_type(kokkos_p, a.extent(0), a.extent(1));
}

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
