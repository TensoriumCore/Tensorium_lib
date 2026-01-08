
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
inline double Dxx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    double inv_dx2 = 1.0 / (dx * dx);
    return (-get(f, i + 2, j, k) + 16.0 * get(f, i + 1, j, k) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i - 1, j, k) - get(f, i - 2, j, k)) *
           (1.0 / 12.0) * inv_dx2;
}

template <typename Field>
inline double Dyy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    double inv_dy2 = 1.0 / (dy * dy);
    return (-get(f, i, j + 2, k) + 16.0 * get(f, i, j + 1, k) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i, j - 1, k) - get(f, i, j - 2, k)) *
           (1.0 / 12.0) * inv_dy2;
}

template <typename Field>
inline double Dzz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    double inv_dz2 = 1.0 / (dz * dz);
    return (-get(f, i, j, k + 2) + 16.0 * get(f, i, j, k + 1) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i, j, k - 1) - get(f, i, j, k - 2)) *
           (1.0 / 12.0) * inv_dz2;
}

template <typename Field>
inline double Dxy(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    return (get(f, i + 1, j + 1, k) - get(f, i + 1, j - 1, k) - get(f, i - 1, j + 1, k) +
            get(f, i - 1, j - 1, k)) /
           (4.0 * dx * dy);
}

template <typename Field>
inline double Dxz(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    return (get(f, i + 1, j, k + 1) - get(f, i + 1, j, k - 1) - get(f, i - 1, j, k + 1) +
            get(f, i - 1, j, k - 1)) /
           (4.0 * dx * dz);
}

template <typename Field>
inline double Dyz(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    return (get(f, i, j + 1, k + 1) - get(f, i, j + 1, k - 1) - get(f, i, j - 1, k + 1) +
            get(f, i, j - 1, k - 1)) /
           (4.0 * dy * dz);
}

// Ajoute ceci dans tensorium_RG::fd

template <typename Field>
inline double Dxy4(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    double sum = 0.0;

    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i + idx[a], j + idx[b], k);
        }
    }

    return sum / (144.0 * dx * dy);
}

template <typename Field>
inline double Dxz4(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    double       sum = 0.0;
    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i + idx[a], j, k + idx[b]);
        }
    }
    return sum / (144.0 * dx * dz);
}

template <typename Field>
inline double Dyz4(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    double       sum = 0.0;
    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i, j + idx[a], k + idx[b]);
        }
    }
    return sum / (144.0 * dy * dz);
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
