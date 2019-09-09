
#ifndef __IMGPROC__CPP__TYPE_UTILS_HPP__
#define __IMGPROC__CPP__TYPE_UTILS_HPP__

#include <cstddef>
#include <type_traits>

namespace type_utils
{
    namespace _impl
    {
        template <size_t I>
        struct index_visit
        {
            template <typename Visitor>
            static inline auto visit(Visitor&& vis, std::size_t idx)
            {
                using Idx = std::integral_constant<std::size_t, I - 1>;
                if (idx == Idx::value)
                    return vis(Idx{});
                else
                    return index_visit<Idx::value>::visit(std::forward<Visitor>(vis), idx);
            }
        };

        template <>
        struct index_visit<1>
        {
            template <typename Visitor>
            static inline auto visit(Visitor&& vis, std::size_t)
            {
                using Idx = std::integral_constant<std::size_t, 0>;
                return vis(Idx{});
            }
        };
    }   // namespace _impl

    template <std::size_t Size, class Visitor>
    constexpr inline auto index_visit(Visitor&& vis, size_t idx)
    {
        return _impl::index_visit<Size>::visit(std::forward<Visitor>(vis), idx);
    }
}   // namespace type_utils

#endif
