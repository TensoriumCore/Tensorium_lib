#pragma once
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace tensorium_RG {

#ifndef TENSORIUM_ALIGN
#    define TENSORIUM_ALIGN 64
#endif

#if defined(__AVX512F__)
#    define TENSORIUM_SIMD_WIDTH_F32 16
#    define TENSORIUM_SIMD_WIDTH_F64 8
#elif defined(__AVX2__)
#    define TENSORIUM_SIMD_WIDTH_F32 8
#    define TENSORIUM_SIMD_WIDTH_F64 4
#else
#    define TENSORIUM_SIMD_WIDTH_F32 4
#    define TENSORIUM_SIMD_WIDTH_F64 2
#endif

template <typename T> inline constexpr size_t simd_width_elems() noexcept {
    if constexpr (std::is_same_v<T, float>)
        return TENSORIUM_SIMD_WIDTH_F32;
    if constexpr (std::is_same_v<T, double>)
        return TENSORIUM_SIMD_WIDTH_F64;
    return 1;
}

template <typename T> inline size_t pad_simd(size_t n) noexcept {
    const size_t w = simd_width_elems<T>();
    if (w <= 1)
        return n;
    return ((n + w - 1) / w) * w;
}

template <typename T> struct AlignedDeleter {
    void operator()(T *p) const noexcept {
        ::operator delete[](p, std::align_val_t(TENSORIUM_ALIGN));
    }
};

template <typename T> using aligned_unique_ptr = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T> inline aligned_unique_ptr<T> aligned_alloc_n(size_t n) {
    return aligned_unique_ptr<T>(
        static_cast<T *>(::operator new[](n * sizeof(T), std::align_val_t(TENSORIUM_ALIGN))));
}

struct GridDims {
    size_t nx, ny, nz;
    size_t ng;
};

template <typename T> struct Strides {
    size_t sx, sy, sz;
    size_t nx_tot, ny_tot, nz_tot;
};

template <typename T> struct Field3D {
    aligned_unique_ptr<T> data;
    Strides<T>            st;

    inline T *ptr() noexcept { return data.get(); }
    inline const T *ptr() const noexcept { return data.get(); }

    inline size_t idx(size_t i, size_t j, size_t k) const noexcept {
        return i * st.sx + j * st.sy + k * st.sz;
    }
};

template <typename T> inline void halo_periodic(Field3D<T> &f, const GridDims &D) {
    const size_t I0 = D.ng, I1 = D.ng + D.nx;
    const size_t J0 = D.ng, J1 = D.ng + D.ny;
    const size_t K0 = D.ng, K1 = D.ng + D.nz;

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t isrc = I1 - 1 - g;
        const size_t idst = I0 - 1 - g;
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k)
                f.ptr()[f.idx(idst, j, k)] = f.ptr()[f.idx(isrc, j, k)];
    }

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t isrc = I0 + g;
        const size_t idst = I1 + g;
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k)
                f.ptr()[f.idx(idst, j, k)] = f.ptr()[f.idx(isrc, j, k)];
    }

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t jsrc = J1 - 1 - g;
        const size_t jdst = J0 - 1 - g;
        for (size_t i = I0; i < I1; ++i)
            for (size_t k = K0; k < K1; ++k)
                f.ptr()[f.idx(i, jdst, k)] = f.ptr()[f.idx(i, jsrc, k)];
    }

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t jsrc = J0 + g;
        const size_t jdst = J1 + g;
        for (size_t i = I0; i < I1; ++i)
            for (size_t k = K0; k < K1; ++k)
                f.ptr()[f.idx(i, jdst, k)] = f.ptr()[f.idx(i, jsrc, k)];
    }

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t ksrc = K1 - 1 - g;
        const size_t kdst = K0 - 1 - g;
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                f.ptr()[f.idx(i, j, kdst)] = f.ptr()[f.idx(i, j, ksrc)];
    }

    for (size_t g = 0; g < D.ng; ++g) {
        const size_t ksrc = K0 + g;
        const size_t kdst = K1 + g;
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                f.ptr()[f.idx(i, j, kdst)] = f.ptr()[f.idx(i, j, ksrc)];
    }
}

enum Sym6 { XX = 0, XY = 1, XZ = 2, YY = 3, YZ = 4, ZZ = 5 };

inline int sym6_index(int i, int j) noexcept {
    if (i > j) {
        int t = i;
        i = j;
        j = t;
    }
    if (i == 0 && j == 0)
        return XX;
    if (i == 0 && j == 1)
        return XY;
    if (i == 0 && j == 2)
        return XZ;
    if (i == 1 && j == 1)
        return YY;
    if (i == 1 && j == 2)
        return YZ;
    return ZZ;
}

template <typename T>
inline void store_sym6(Field3D<T> *f6, size_t idx, T xx, T xy, T xz, T yy, T yz, T zz) {
    f6[XX].ptr()[idx] = xx;
    f6[XY].ptr()[idx] = xy;
    f6[XZ].ptr()[idx] = xz;
    f6[YY].ptr()[idx] = yy;
    f6[YZ].ptr()[idx] = yz;
    f6[ZZ].ptr()[idx] = zz;
}

template <typename T>
inline void load_sym6(Field3D<T> *f6, size_t idx, T &xx, T &xy, T &xz, T &yy, T &yz, T &zz) {
    xx = f6[XX].ptr()[idx];
    xy = f6[XY].ptr()[idx];
    xz = f6[XZ].ptr()[idx];
    yy = f6[YY].ptr()[idx];
    yz = f6[YZ].ptr()[idx];
    zz = f6[ZZ].ptr()[idx];
}

template <typename T> inline Field3D<T> make_field(const Strides<T> &st) {
    Field3D<T> f;
    f.st = st;
    const size_t N = st.nx_tot * st.ny_tot * st.nz_tot;
    f.data = aligned_alloc_n<T>(N);
    return f;
}

struct BoundaryPeriodic {
    template <typename T> static inline void apply(Field3D<T> &f, const GridDims &D) {
        halo_periodic(f, D);
    }
};

template <typename T> inline void halo_clamp_full(Field3D<T> &f, const GridDims &D) {
    const size_t nxT = D.nx + 2 * D.ng;
    const size_t nyT = D.ny + 2 * D.ng;
    const size_t nzT = D.nz + 2 * D.ng;

    const size_t I0 = D.ng, I1 = D.ng + D.nx;
    const size_t J0 = D.ng, J1 = D.ng + D.ny;
    const size_t K0 = D.ng, K1 = D.ng + D.nz;

    auto clamp_i = [&](size_t i) -> size_t {
        if (i < I0)
            return I0;
        if (i >= I1)
            return I1 - 1;
        return i;
    };
    auto clamp_j = [&](size_t j) -> size_t {
        if (j < J0)
            return J0;
        if (j >= J1)
            return J1 - 1;
        return j;
    };
    auto clamp_k = [&](size_t k) -> size_t {
        if (k < K0)
            return K0;
        if (k >= K1)
            return K1 - 1;
        return k;
    };

    for (size_t i = 0; i < nxT; ++i)
        for (size_t j = 0; j < nyT; ++j)
            for (size_t k = 0; k < nzT; ++k) {
                const bool inInterior =
                    (i >= I0 && i < I1) && (j >= J0 && j < J1) && (k >= K0 && k < K1);
                if (inInterior)
                    continue;

                const size_t is = clamp_i(i);
                const size_t js = clamp_j(j);
                const size_t ks = clamp_k(k);

                f.ptr()[f.idx(i, j, k)] = f.ptr()[f.idx(is, js, ks)];
            }
}
struct BoundaryClamp {
    template <typename T> static inline void apply(Field3D<T> &f, const GridDims &D) {
        halo_clamp_full(f, D);
    }
};

template <typename T> inline T sym6_get(const Field3D<T> *f6, size_t idx, int i, int j) noexcept {
    if (i > j) {
        int t = i;
        i = j;
        j = t;
    }
    if (i == 0 && j == 0)
        return f6[XX].ptr()[idx];
    if (i == 0 && j == 1)
        return f6[XY].ptr()[idx];
    if (i == 0 && j == 2)
        return f6[XZ].ptr()[idx];
    if (i == 1 && j == 1)
        return f6[YY].ptr()[idx];
    if (i == 1 && j == 2)
        return f6[YZ].ptr()[idx];
    return f6[ZZ].ptr()[idx];
}

template <typename T>
inline T sym6_inv_get(const Field3D<T> *f6, size_t idx, int i, int j) noexcept {
    return sym6_get(f6, idx, i, j);
}

} // namespace tensorium_RG
