
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
  auto kokkos_p = to_kokkos_pointer(a.data());
  using view_type = Kokkos::View<
    decltype(kokkos_p)*, typename LayoutMapper<Layout>::type
    >;
  return view_type(kokkos_p, a.extent(0), a.extent(1));
}

/*
  partially specialize for when the mdspan has transposed layout.

  Note that here we basically neglect the transposition
  and convert the "nested/original" mdspan.
  This way, the task of accounting the transposition is
  handed over to the algorithm impl. The reason for this is the following.
  Suppose that one has: A, B, C : mdspans
  and wants to do:  C = A^T B
  One would do this by calling:
    AT = std::experimental::linalg::transposed(A)
    matrix_product(kokkos_exec<>, AT, B, C)

  Our Kokkos impl would then see:
   matrix_product(kokkos_exec<>, Ain, Bin, Cin)

  so that Ain would reference AT, Bin would reference B, etc.
  Inside our impl we convert the mdspan arguments o kokkos views.
  Suppose two scenarios:
  (1) when converting Ain to a view (Ain_view) we account for the transposition.
      In this case, Ain_view would already carry the effect of the transpose.
      To call the KK impl, we would then need do:
          KokkosBlas::gemm("N", "N", alpha, Ain_view, Bin_view, beta, Cin_view);
      Note that here we do NOT pass "T" as first arg because Ain_view
      already carries the tranposition.

  (2) when converting Ain to a view (Ain_view) we do NOT account for the transposition.
      To call the KK impl, we need to explicitly express the transpose:
          KokkosBlas::gemm("T", "N", alpha, Ain_view, Bin_view, beta, Cin_view);
      In this case, we need "T" because Ain_view would NOT already
      carry the effect of the transpose.

  Here we take option 2 to avoid allways calling the fallback impl.
*/
template<
  class ElementType,
  std::experimental::extents<>::size_type ext0,
  std::experimental::extents<>::size_type ext1,
  class NestedLayout,
  class Accessor>
auto mdspan_to_view(std::experimental::mdspan<
		    ElementType,
		    std::experimental::extents<ext0, ext1>,
		    std::experimental::linalg::layout_transpose<NestedLayout>,
		    Accessor
		    > a)
{
  auto kokkos_p = to_kokkos_pointer(a.data());
  using view_type = Kokkos::View<
    decltype(kokkos_p)*, typename LayoutMapper<NestedLayout>::type
    >;
  // note that here a is the transposed mdspan so to get the
  // correct extents of the original mdpsan we need to invert indices
  return view_type(kokkos_p, a.extent(1), a.extent(0));
}

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
