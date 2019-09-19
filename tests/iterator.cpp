#include <experimental/linalg>
#include <experimental/mdspan>
#include <algorithm>
#include <iterator>
#include <limits>
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
    using reference = typename mdspan_t::reference;
    using difference_type = typename mdspan_t::difference_type;

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

    iterator begin() {
      return iterator(x_);
    }

    iterator end() {
      return iterator(x_.extent(0));
    }

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
    return iterator (x);
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
    return iterator (x, x.extent(0));
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

    auto A_col0 = subspan(A, all, 0);
    EXPECT_TRUE( A_col0.stride(0) != 1 );

    const bool ok_col = testRotateSort(A_col0);
    ASSERT_TRUE( ok_col );

    auto A_row0 = subspan(A, 0, all);
    EXPECT_TRUE( A_row0.stride(0) == 1 );

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
