
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_ADD_HPP_

namespace KokkosKernelsSTD {

template <class ViewXType, class ViewYType, class ViewZType>
struct MyTmpAddFunctorRank1
{
  ViewXType m_x;
  ViewYType m_y;
  ViewZType m_z;

  MyTmpAddFunctorRank1() = delete;
  MyTmpAddFunctorRank1(const ViewXType x,
		       const ViewYType y,
		       const ViewZType z)
    : m_x(x), m_y(y), m_z(z){}

  KOKKOS_INLINE_FUNCTION
  void operator()(std::size_t i) const{
    m_z(i) = m_x(i) + m_y(i);
  }
};

template <class ViewXType, class ViewYType, class ViewZType>
struct MyTmpAddFunctorRank2
{
  const std::size_t m_numCols;
  ViewXType m_x;
  ViewYType m_y;
  ViewZType m_z;

  MyTmpAddFunctorRank2() = delete;
  MyTmpAddFunctorRank2(const ViewXType x,
		  const ViewYType y,
		  const ViewZType z)
    : m_numCols(x.extent(1)), m_x(x), m_y(y), m_z(z){}

  KOKKOS_INLINE_FUNCTION
  void operator()(std::size_t i) const
  {
    for (std::size_t k = 0; k < m_numCols; ++k)
    {
      m_z(i,k) = m_x(i,k) + m_y(i,k);
    }
  }
};

template<class ElementType_x,
         std::experimental::extents<>::size_type ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ... ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         std::experimental::extents<>::size_type ... ext_z,
         class Layout_z,
         class Accessor_z>
  requires (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z))
void add(
  kokkos_exec<>,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x ...>, Layout_x, Accessor_x> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y ...>, Layout_y, Accessor_y> y,
  std::experimental::mdspan<ElementType_z, std::experimental::extents<ext_z ...>, Layout_z, Accessor_z> z)
{
  static_assert(z.rank() <= 2);

  auto x_view = Impl::mdspan_to_view(x);
  using x_view_type = decltype(x_view);

  auto y_view = Impl::mdspan_to_view(y);
  using y_view_type = decltype(y_view);

  auto z_view = Impl::mdspan_to_view(z);
  using z_view_type = decltype(z_view);

  // change this after adding the correct impl to KK
  if constexpr (z.rank() == 1) {
    using func_t = MyTmpAddFunctorRank1<x_view_type, y_view_type, z_view_type>;
    func_t F(x_view, y_view, z_view);
    Kokkos::parallel_for("stdBLAS::KokkosKernelsSTD_add_rank_1", x_view.extent(0), F);
  }
  else if constexpr (z.rank() == 2) {
    using func_t = MyTmpAddFunctorRank2<x_view_type, y_view_type, z_view_type>;
    func_t F(x_view, y_view, z_view);
    Kokkos::parallel_for("stdBLAS::KokkosKernelsSTD_add_rank_2", x_view.extent(0), F);
  }
}

}
#endif
