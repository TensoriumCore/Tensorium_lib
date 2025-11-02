#pragma once 
#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <memory>
#include <type_traits>

namespace tensorium_RG {

#ifndef TENSORIUM_ALIGN
#define TENSORIUM_ALIGN 64
#endif

#ifndef TENSORIUM_SIMD_WIDTH_F32
#if defined(__AVX512F__)
#define TENSORIUM_SIMD_WIDTH_F32 16
#elif defined(__AVX2__)
#define TENSORIUM_SIMD_WIDTH_F32 8
#else
#define TENSORIUM_SIMD_WIDTH_F32 4
#endif
#endif

template <typename T> inline size_t pad_simd(size_t n) {
  const size_t w =
      (std::is_same<T, float>::value ? TENSORIUM_SIMD_WIDTH_F32 : 8);
  return ((n + w - 1) / w) * w;
}

template <typename T> struct AlignedDeleter {
  void operator()(T *p) const noexcept {
    ::operator delete[](p, std::align_val_t(TENSORIUM_ALIGN));
  }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T> aligned_unique_ptr<T> aligned_alloc_n(size_t n) {
  return aligned_unique_ptr<T>(static_cast<T *>(
      ::operator new[](n * sizeof(T), std::align_val_t(TENSORIUM_ALIGN))));
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
  Strides<T> st;

  inline T *ptr() const noexcept { return data.get(); }
  inline size_t idx(size_t i, size_t j, size_t k) const noexcept {
    return i * st.sx + j * st.sy + k * st.sz;
  }
};

template <typename T>
void halo_periodic(Field3D<T> &f, const GridDims &D, const Strides<T> &st) {
  const size_t I0 = D.ng, I1 = D.ng + D.nx;
  const size_t J0 = D.ng, J1 = D.ng + D.ny;
  const size_t K0 = D.ng, K1 = D.ng + D.nz;

  for (size_t g = 0; g < D.ng; ++g) {
    size_t isrc = I1 - 1 - g;
    size_t idst = I0 - 1 - g;
    for (size_t j = J0; j < J1; ++j)
      for (size_t k = K0; k < K1; ++k)
        f.ptr()[idst * st.sx + j * st.sy + k] =
            f.ptr()[isrc * st.sx + j * st.sy + k];
  }

  for (size_t g = 0; g < D.ng; ++g) {
    size_t isrc = I0 + g;
    size_t idst = I1 + g;
    for (size_t j = J0; j < J1; ++j)
      for (size_t k = K0; k < K1; ++k)
        f.ptr()[idst * st.sx + j * st.sy + k] =
            f.ptr()[isrc * st.sx + j * st.sy + k];
  }

  for (size_t g = 0; g < D.ng; ++g) {
    size_t jsrc = J1 - 1 - g, jdst = J0 - 1 - g;
    for (size_t i = I0; i < I1; ++i)
      for (size_t k = K0; k < K1; ++k)
        f.ptr()[i * st.sx + jdst * st.sy + k] =
            f.ptr()[i * st.sx + jsrc * st.sy + k];
  }
  for (size_t g = 0; g < D.ng; ++g) {
    size_t jsrc = J0 + g, jdst = J1 + g;
    for (size_t i = I0; i < I1; ++i)
      for (size_t k = K0; k < K1; ++k)
        f.ptr()[i * st.sx + jdst * st.sy + k] =
            f.ptr()[i * st.sx + jsrc * st.sy + k];
  }

  for (size_t g = 0; g < D.ng; ++g) {
    size_t ksrc = K1 - 1 - g, kdst = K0 - 1 - g;
    for (size_t i = I0; i < I1; ++i)
      for (size_t j = J0; j < J1; ++j)
        f.ptr()[i * st.sx + j * st.sy + kdst] =
            f.ptr()[i * st.sx + j * st.sy + ksrc];
  }
  for (size_t g = 0; g < D.ng; ++g) {
    size_t ksrc = K0 + g, kdst = K1 + g;
    for (size_t i = I0; i < I1; ++i)
      for (size_t j = J0; j < J1; ++j)
        f.ptr()[i * st.sx + j * st.sy + kdst] =
            f.ptr()[i * st.sx + j * st.sy + ksrc];
  }
}

enum Sym6 { XX = 0, XY = 1, XZ = 2, YY = 3, YZ = 4, ZZ = 5 };

template <typename T> class BSSNGridSoA {
public:
  GridDims dims;
  Strides<T> st;

  Field3D<T> alpha, chi;

  Field3D<T> beta[3], tildeGamma[3], contractedGamma[3];

  Field3D<T> gamma_ij[6], gamma_ij_inv[6];
  Field3D<T> gamma_tilde[6], gamma_tilde_inv[6];
  Field3D<T> A_tilde[6], K_ij[6];

  Field3D<T> d_beta[3][3];
  Field3D<T> d_gamma[6][3];

  T dx, dy, dz;

  BSSNGridSoA(size_t nx, size_t ny, size_t nz, size_t ng, T dx_, T dy_, T dz_)
      : dims{nx, ny, nz, ng}, dx(dx_), dy(dy_), dz(dz_) {
    const size_t nx_tot = pad_simd<T>(nx + 2 * ng);
    const size_t ny_tot = ny + 2 * ng;
    const size_t nz_tot = pad_simd<T>(nz + 2 * ng);

    st.nx_tot = nx_tot;
    st.ny_tot = ny_tot;
    st.nz_tot = nz_tot;
    st.sz = 1;
    st.sy = nz_tot;
    st.sx = ny_tot * nz_tot;

    auto alloc_field = [&](Field3D<T> &f) {
      const size_t N = nx_tot * ny_tot * nz_tot;
      f.data = aligned_alloc_n<T>(N);
      f.st = st;
    };

    alloc_field(alpha);
    alloc_field(chi);
    for (int i = 0; i < 3; ++i) {
      alloc_field(beta[i]);
      alloc_field(tildeGamma[i]);
      alloc_field(contractedGamma[i]);
    }
    for (int s = 0; s < 6; ++s) {
      alloc_field(gamma_ij[s]);
      alloc_field(gamma_ij_inv[s]);
      alloc_field(gamma_tilde[s]);
      alloc_field(gamma_tilde_inv[s]);
      alloc_field(A_tilde[s]);
      alloc_field(K_ij[s]);
    }
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        alloc_field(d_beta[i][j]);
    for (int s = 0; s < 6; ++s)
      for (int i = 0; i < 3; ++i)
        alloc_field(d_gamma[s][i]);
  }

  inline void domain_bounds(size_t &i0, size_t &i1, size_t &j0, size_t &j1,
                            size_t &k0, size_t &k1) const noexcept {
    i0 = dims.ng;
    i1 = dims.ng + dims.nx;
    j0 = dims.ng;
    j1 = dims.ng + dims.ny;
    k0 = dims.ng;
    k1 = dims.ng + dims.nz;
  }

  T x0 = 0, y0 = 0, z0 = 0;
  inline void coords(size_t i, size_t j, size_t k, T &x, T &y,
                     T &z) const noexcept {
    x = x0 + (i - dims.ng) * dx;
    y = y0 + (j - dims.ng) * dy;
    z = z0 + (k - dims.ng) * dz;
  }
};

template <typename T>
inline void store_sym6(Field3D<T> *f6, size_t idx, T xx, T xy, T xz, T yy, T yz,
                       T zz) {
  f6[XX].ptr()[idx] = xx;
  f6[XY].ptr()[idx] = xy;
  f6[XZ].ptr()[idx] = xz;
  f6[YY].ptr()[idx] = yy;
  f6[YZ].ptr()[idx] = yz;
  f6[ZZ].ptr()[idx] = zz;
}

template <typename T>
inline void load_sym6(Field3D<T> *f6, size_t idx, T &xx, T &xy, T &xz, T &yy,
                      T &yz, T &zz) {
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
  template <typename T>
  static void apply(Field3D<T> &f, const GridDims &D, const Strides<T> &st) {
    halo_periodic(f, D, st);
  }
};

struct BoundaryClamp {
  template <typename T>
  static void apply(Field3D<T> &f, const GridDims &D, const Strides<T> &st) {
    const size_t I0 = D.ng, I1 = D.ng + D.nx, J0 = D.ng, J1 = D.ng + D.ny,
                 K0 = D.ng, K1 = D.ng + D.nz;
    for (size_t g = 0; g < D.ng; ++g) {
      size_t idst = I0 - 1 - g, isrc = I0;
      for (size_t j = J0; j < J1; ++j)
        for (size_t k = K0; k < K1; ++k)
          f.ptr()[idst * st.sx + j * st.sy + k] =
              f.ptr()[isrc * st.sx + j * st.sy + k];
    }
    for (size_t g = 0; g < D.ng; ++g) {
      size_t idst = I1 + g, isrc = I1 - 1;
      for (size_t j = J0; j < J1; ++j)
        for (size_t k = K0; k < K1; ++k)
          f.ptr()[idst * st.sx + j * st.sy + k] =
              f.ptr()[isrc * st.sx + j * st.sy + k];
    }
    for (size_t g = 0; g < D.ng; ++g) {
      size_t jdst = J0 - 1 - g, jsrc = J0;
      for (size_t i = I0; i < I1; ++i)
        for (size_t k = K0; k < K1; ++k)
          f.ptr()[i * st.sx + jdst * st.sy + k] =
              f.ptr()[i * st.sx + jsrc * st.sy + k];
    }
    for (size_t g = 0; g < D.ng; ++g) {
      size_t jdst = J1 + g, jsrc = J1 - 1;
      for (size_t i = I0; i < I1; ++i)
        for (size_t k = K0; k < K1; ++k)
          f.ptr()[i * st.sx + jdst * st.sy + k] =
              f.ptr()[i * st.sx + jsrc * st.sy + k];
    }
    for (size_t g = 0; g < D.ng; ++g) {
      size_t kdst = K0 - 1 - g, ksrc = K0;
      for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
          f.ptr()[i * st.sx + j * st.sy + kdst] =
              f.ptr()[i * st.sx + j * st.sy + ksrc];
    }
    for (size_t g = 0; g < D.ng; ++g) {
      size_t kdst = K1 + g, ksrc = K1 - 1;
      for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
          f.ptr()[i * st.sx + j * st.sy + kdst] =
              f.ptr()[i * st.sx + j * st.sy + ksrc];
    }
  }
};

template <class Boundary, typename T>
inline void apply_halos_grid(BSSNGridSoA<T> &G) {
  auto &D = G.dims;
  auto &st = G.st;
  Boundary::apply(G.alpha, D, st);
  Boundary::apply(G.chi, D, st);
  for (int c = 0; c < 3; ++c) {
    Boundary::apply(G.beta[c], D, st);
    Boundary::apply(G.tildeGamma[c], D, st);
    Boundary::apply(G.contractedGamma[c], D, st);
  }
  for (int s = 0; s < 6; ++s) {
    Boundary::apply(G.gamma_ij[s], D, st);
    Boundary::apply(G.gamma_ij_inv[s], D, st);
    Boundary::apply(G.gamma_tilde[s], D, st);
    Boundary::apply(G.gamma_tilde_inv[s], D, st);
    Boundary::apply(G.A_tilde[s], D, st);
    Boundary::apply(G.K_ij[s], D, st);
  }
}
} // namespace tensorium_RG
