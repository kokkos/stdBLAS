#include <experimental/linalg>
#include <experimental/mdspan>
#include <algorithm>
#include <iterator>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::all;
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::layout_left;
  using std::experimental::layout_right;
  using std::experimental::layout_stride; // does compile
  using std::experimental::basic_mdspan;
  using std::experimental::subspan;

  template<class ElementType,
           class Extents,
           class Layout,
           class Accessor>
  class MdspanRandomAccessIterator {};

  template<class ElementType,
           ptrdiff_t extent,
           class Layout,
           class Accessor>
  class MdspanRandomAccessIterator<
    ElementType, extents<extent>, Layout, Accessor> :
    public std::iterator<
      std::random_access_iterator_tag,   // iterator_category
      ElementType, // value_type
      typename basic_mdspan<ElementType,
                            extents<extent>,
                            Layout, Accessor>::difference_type,
      typename basic_mdspan<ElementType,
                            extents<extent>,
                            Layout, Accessor>::pointer,
      typename basic_mdspan<ElementType,
                            extents<extent>,
                            Layout, Accessor>::reference>
  {
  public:
    using extents_t = extents<extent>;
    using mdspan_t = basic_mdspan<
      ElementType, extents_t, Layout, Accessor>;
    using iterator = MdspanRandomAccessIterator<
      ElementType, extents_t, Layout, Accessor>;
    using difference_type = typename mdspan_t::difference_type;
    using reference = typename mdspan_t::reference;
    using pointer = typename mdspan_t::pointer;

    explicit MdspanRandomAccessIterator(mdspan_t x) : x_(x) {}
    explicit MdspanRandomAccessIterator(mdspan_t x,
                               ptrdiff_t current_index) :
      x_(x), current_index_ (current_index) {}

    iterator& operator++() {
      ++current_index_;
      return *this;
    }

    iterator operator++(int) {
      return iterator (x_, current_index_+1);
    }

    iterator& operator--() {
      --current_index_;
      return *this;
    }

    iterator operator--(int) {
      return iterator (x_, current_index_-1);
    }

    iterator operator+(difference_type n) const {
      return iterator (x_, current_index_ + n);
    }

    iterator operator-(difference_type n) const {
      return iterator (x_, current_index_ - n);
    }

    // iterator operator+(iterator it) const {
    //   return iterator (x_, current_index_ + it.current_index_);
    // }

    difference_type operator-(iterator it) const {
      return current_index_ - it.current_index_;
    }

    bool operator==(iterator other) const {
      return current_index_ == other.current_index_ &&
        x_.data() == other.x_.data();
    }

    bool operator!=(iterator other) const {
      return current_index_ != other.current_index_ ||
        x_.data() != other.x_.data();
    }

    bool operator<(iterator other) const {
      return current_index_ < other.current_index_;
    }

    reference operator*() const {
      return x_(current_index_);
    }

    pointer operator->() const {
      return x_.accessor().
        offset(x_.data(), x_.mapping()(current_index_));
    }

    // iterator begin() {
    //   return iterator(x_);
    // }

    // iterator end() {
    //   return iterator(x_.extent(0));
    // }

  private:
    mdspan_t x_;
    ptrdiff_t current_index_ = 0;
  };

  template<class ElementType,
           class Extents,
           class Layout,
           class Accessor>
  MdspanRandomAccessIterator<ElementType, Extents, Layout, Accessor>
  begin(basic_mdspan<ElementType, Extents, Layout, Accessor> x)
  {
    using iterator =
      MdspanRandomAccessIterator<ElementType, Extents, Layout, Accessor>;
    return iterator(x);
  }

  template<class ElementType,
           class Extents,
           class Layout,
           class Accessor>
  MdspanRandomAccessIterator<ElementType, Extents, Layout, Accessor>
  end(basic_mdspan<ElementType, Extents, Layout, Accessor> x)
  {
    using iterator =
      MdspanRandomAccessIterator<ElementType, Extents, Layout, Accessor>;
    return iterator(x, x.extent(0));
  }

  template<class SpanType>
  constexpr bool testMdspanIterator_LegacyIterator_StaticConcept()
  {
    using element_type = typename SpanType::element_type;
    using extents_type = typename SpanType::extents_type;
    using layout_type = typename SpanType::layout_type;
    using accessor_type = typename SpanType::accessor_type;

    using mdspan_type = basic_mdspan<
      element_type, extents_type, layout_type, accessor_type>;
    static_assert(mdspan_type::rank() == 1);
    using pointer = typename mdspan_type::pointer;

    mdspan_type x(nullptr, 0);
    using iterator = decltype(begin(x));
    static_assert(std::is_same_v<iterator, decltype(end(x))>);

    static_assert(std::is_copy_constructible_v<iterator>);
    static_assert(std::is_copy_assignable_v<iterator>);
    static_assert(std::is_destructible_v<iterator>);
    static_assert(std::is_swappable_v<iterator>); // C++17

    static_assert(std::is_same_v<
      typename std::iterator_traits<iterator>::value_type,
      element_type>);
    static_assert(std::is_same_v<
      typename std::iterator_traits<iterator>::difference_type,
      typename mdspan_type::difference_type>);
    static_assert(std::is_same_v<
      typename std::iterator_traits<iterator>::reference,
      typename mdspan_type::reference>);
    static_assert(std::is_same_v<
      typename std::iterator_traits<iterator>::pointer,
      typename mdspan_type::pointer>);

    using reference = typename mdspan_type::reference;

    // This just needs to exist.
    using iterator_category =
      std::iterator_traits<iterator>::iterator_category;

    using ref_t = decltype(*begin(x));
    static_assert(std::is_same_v<ref_t, reference>);
    using iter_plusplus_t =
      typename std::decay<decltype(++begin(x))>::type;
    static_assert(std::is_same_v<iter_plusplus_t, iterator>);

    return true;
  }

  template<class ElementType>
  struct real_traits {
    using type = ElementType;
  };
  template<class RealType>
  struct real_traits<std::complex<RealType>> {
    using type = RealType;
  };

  template<class ElementType>
  using real_t = typename real_traits<ElementType>::type;

  template<class ElementType,
           ptrdiff_t extent,
           class Layout,
           class Accessor>
  bool testRotateSort (
    basic_mdspan<ElementType, extents<extent>, Layout, Accessor> x)
  {
    using mdspan_t = basic_mdspan<
      ElementType, extents<extent>, Layout, Accessor>;
    using scalar_t = ElementType;

    const ptrdiff_t dim = x.extent(0);
    for (ptrdiff_t k = 0; k < dim; ++k) {
      x(k) = scalar_t(real_t<scalar_t>(k+1));
    }
    std::rotate(begin(x), begin(x) + 1, end(x));
    std::sort(begin(x), end(x));

    for (ptrdiff_t k = 0; k < dim; ++k) {
      const ptrdiff_t km1 = (k + dim) % dim;
      if (x(km1) != scalar_t(real_t<scalar_t>(k+1))) {
        return false;
      }
    }
    return true;
  }

  TEST(mdspan_iterators, random_access)
  {
    using real_t = double;
    using scalar_t = real_t;
    //using layout_t = layout_stride; // doesn't compile; why?
    using layout_t = layout_right;
    using extents_t = extents<dynamic_extent, dynamic_extent>;
    using matrix_t = basic_mdspan<scalar_t, extents_t, layout_t>;

    constexpr size_t dim(5);
    constexpr size_t storageSize(dim * dim);

    std::vector<scalar_t> storage(storageSize);
    matrix_t A(storage.data(), dim, dim);
    EXPECT_TRUE( A.stride(0) != 1 );
    EXPECT_TRUE( A.stride(1) == 1 );

    for (ptrdiff_t i = 0; i < dim; ++i) {
      for (ptrdiff_t j = 0; j < dim; ++j) {
        A(i,j) = scalar_t(real_t(j+1)) +
          scalar_t(real_t(A.stride(0)*(i+1)));
      }
    }

    ////////////////////////////////////////////////////////////
    // Test layout_stride
    ////////////////////////////////////////////////////////////

    auto A_col0 = subspan(A, all, 0);
    EXPECT_TRUE( A_col0.stride(0) != 1 );
    // This works only if A is layout_right
    static_assert(! decltype(A_col0)::is_always_contiguous());
    constexpr bool col0_test =
      testMdspanIterator_LegacyIterator_StaticConcept<decltype(A_col0)>();
    static_assert(col0_test);

    {
      auto the_beg = begin(A_col0);
      auto the_end = end(A_col0);
      ASSERT_TRUE( dim == 0 || the_beg != the_end );

      {
        ptrdiff_t k = 0;
        auto it = the_beg;
        for ( ; it != the_end; ++it, ++k) { // test prefix ++
          ASSERT_TRUE( *it == A_col0(k) );
          ASSERT_TRUE( *it == A(k,0) );
          const auto expected_val = scalar_t(real_t(1)) +
            scalar_t(real_t(A.stride(0)*(k+1)));
          ASSERT_TRUE( *it == expected_val );
          ASSERT_TRUE( *(it.operator->()) == expected_val );
        }
        ASSERT_TRUE( it == the_end );
      }

      {
        ptrdiff_t k = 0;
        auto it = the_beg;
        while (it != the_end) {
          ASSERT_TRUE( *it == A_col0(k) );
          ASSERT_TRUE( *it == A(k,0) );
          const auto expected_val = scalar_t(real_t(1)) +
            scalar_t(real_t(A.stride(0)*(k+1)));
          ASSERT_TRUE( *it == expected_val );
          ASSERT_TRUE( *(it.operator->()) == expected_val );
          it = it++; // test postfix ++
          ++k;
        }
        ASSERT_TRUE( it == the_end );
      }
    }

    const bool ok_col = testRotateSort(A_col0);
    ASSERT_TRUE( ok_col );

    for (ptrdiff_t i = 0; i < dim; ++i) {
      for (ptrdiff_t j = 0; j < dim; ++j) {
        A(i,j) = scalar_t(real_t(j+1)) +
          scalar_t(real_t(A.stride(0)*(i+1)));
      }
    }

    ////////////////////////////////////////////////////////////
    // Test layout_right
    ////////////////////////////////////////////////////////////

    auto A_row0 = subspan(A, 0, all);
    EXPECT_TRUE( A_row0.stride(0) == 1 );

    // This works only if A is layout_right.  We need this because we
    // want to test LegacyContiguousIterator below.
    static_assert(decltype(A_row0)::is_always_contiguous());

    constexpr bool row0_test =
      testMdspanIterator_LegacyIterator_StaticConcept<decltype(A_row0)>();
    static_assert(row0_test);

    {
      auto the_beg = begin(A_row0);
      auto the_end = end(A_row0);
      ASSERT_TRUE( dim == 0 || the_beg != the_end );

      {
        ptrdiff_t k = 0;
        auto it = the_beg;
        for ( ; it != the_end; ++it, ++k) { // test prefix ++
          ASSERT_TRUE( *it == A_row0(k) );
          ASSERT_TRUE( *it == A(0,k) );
          const auto expected_val = scalar_t(real_t(k+1)) +
            scalar_t(real_t(A.stride(0)));
          ASSERT_TRUE( *it == expected_val );
          ASSERT_TRUE( *(it.operator->()) == expected_val );

          // *(a + k) is equivalent to *(std::addressof(*a) + k)
          ASSERT_TRUE( *(the_beg + k) == *(std::addressof(*the_beg) + k) );
        }
        ASSERT_TRUE( it == the_end );
      }

      {
        ptrdiff_t k = 0;
        auto it = the_beg;
        while (it != the_end) {
          ASSERT_TRUE( *it == A_row0(k) );
          ASSERT_TRUE( *it == A(0,k) );
          const auto expected_val = scalar_t(real_t(k+1)) +
            scalar_t(real_t(A.stride(0)));
          ASSERT_TRUE( *it == expected_val );
          ASSERT_TRUE( *(it.operator->()) == expected_val );

          it = it++; // test postfix ++
          ++k;
        }
        ASSERT_TRUE( it == the_end );
      }
    }

    const bool ok_row = testRotateSort(A_row0);
    ASSERT_TRUE( ok_row );
  }

  // TEST(mdspan_iterators, contiguous)
  // {
  //   using real_t = double;
  //   using scalar_t = real_t;
  //   using layout_t = layout_left;
  //   using extents_t = extents<dynamic_extent>;
  //   using vector_t = basic_mdspan<scalar_t, extents_t, layout_t>;

  //   constexpr size_t vectorSize(10);
  //   constexpr size_t storageSize(10);
  //   std::vector<scalar_t> storage(storageSize);
  //   vector_t x(storage.data(), vectorSize);
  // }
}
