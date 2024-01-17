//  Copyright (c) 2022 Hartmut Kaiser

namespace hpxtesting {

template <class T>
auto create_stdvector_and_copy(T sourceView)
{
    static_assert(sourceView.rank() == 1);

    using value_type = typename T::value_type;
    using res_t = std::vector<value_type>;

    res_t result(sourceView.extent(0));
    for (std::size_t i = 0; i < sourceView.extent(0); ++i)
    {
        result[i] = sourceView(i);
    }

    return result;
}

}    // namespace hpxtesting
