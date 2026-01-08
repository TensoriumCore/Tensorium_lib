
#pragma once

#include <cstddef>

namespace tensorium_RG::fd {

enum Axis { X = 0, Y = 1, Z = 2 };

template <typename Field> inline double get(const Field &f, size_t i, size_t j, size_t k) {
    return f.ptr()[f.idx(i, j, k)];
}

template <typename Field>
inline double Dx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    return (-get(f, i + 2, j, k) + 8.0 * get(f, i + 1, j, k) - 8.0 * get(f, i - 1, j, k) +
            get(f, i - 2, j, k)) /
           (12.0 * dx);
}

template <typename Field>
inline double Dy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    return (-get(f, i, j + 2, k) + 8.0 * get(f, i, j + 1, k) - 8.0 * get(f, i, j - 1, k) +
            get(f, i, j - 2, k)) /
           (12.0 * dy);
}

template <typename Field>
inline double Dz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    return (-get(f, i, j, k + 2) + 8.0 * get(f, i, j, k + 1) - 8.0 * get(f, i, j, k - 1) +
            get(f, i, j, k - 2)) /
           (12.0 * dz);
}

template <typename Field>
inline double D(const Field &f, size_t i, size_t j, size_t k, Axis a, double dx, double dy,
                double dz) {
    if (a == X)
        return Dx(f, i, j, k, dx);
    if (a == Y)
        return Dy(f, i, j, k, dy);
    return Dz(f, i, j, k, dz);
}

} // namespace tensorium_RG::fd
