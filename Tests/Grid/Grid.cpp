#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/BSSNInitialData.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/BSSNMetricSplit.hpp"
#include "../includes/Tensorium_Grid/GridSetup.hpp"
#include "../test.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

using namespace tensorium_RG;

static void print_strides(const Strides<double> &st) {
    printf("Strides:\n");
    printf("  sx = %zu\n", st.sx);
    printf("  sy = %zu\n", st.sy);
    printf("  sz = %zu\n", st.sz);
    printf("  nx_tot = %zu\n", st.nx_tot);
    printf("  ny_tot = %zu\n", st.ny_tot);
    printf("  nz_tot = %zu\n", st.nz_tot);
}

static void print_dims(const GridDims &d) {
    printf("GridDims:\n");
    printf("  nx = %zu, ny = %zu, nz = %zu, ng = %zu\n", d.nx, d.ny, d.nz, d.ng);
}

static inline double tag_val(size_t i, size_t j, size_t k) {
    return 1000000.0 * double(i) + 1000.0 * double(j) + double(k);
}

static void fill_interior(Field3D<double> &f, const BSSNGridSoA<double> &G) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k)
                f.ptr()[f.idx(i, j, k)] = tag_val(i - I0, j - J0, k - K0);
}

template <typename T> void export_chi_slice(const BSSNGridSoA<T> &G, const std::string &filename) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    size_t k_mid = K0 + G.dims.nz / 2;

    std::ofstream file(filename);
    file << "x,y,chi\n";

    for (size_t i = I0; i < I1; ++i) {
        for (size_t j = J0; j < J1; ++j) {
            T x, y, z;
            G.coords(i, j, k_mid, x, y, z);

            size_t id = G.chi.idx(i, j, k_mid);
            file << x << "," << y << "," << G.chi.ptr()[id] << "\n";
        }
    }
    file.close();
    printf("Slice exported to %s\n", filename.c_str());
}
#include <algorithm>
#include <cmath>

template <typename T>
void export_log_chi_slice(const BSSNGridSoA<T> &G, const std::string &filename) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    size_t k_mid = I0 + (G.dims.nz / 2);

    std::ofstream file(filename);
    file << "i,j,log_chi\n";

    for (size_t i = 0; i < G.st.nx_tot; ++i) {
        for (size_t j = 0; j < G.st.ny_tot; ++j) {
            size_t id = G.chi.idx(i, j, k_mid);
            T      val = G.chi.ptr()[id];

            T log_val = std::log10(std::max(val, T(1e-20)));

            file << i << "," << j << "," << log_val << "\n";
        }
    }
    file.close();
}

template <typename T>
void export_alpha_slice(const BSSNGridSoA<T> &G, const std::string &filename) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    size_t k_mid = I0 + (G.dims.nz / 2);

    std::ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    file << "i,j,alpha\n";

    for (size_t i = 0; i < G.st.nx_tot; ++i) {
        for (size_t j = 0; j < G.st.ny_tot; ++j) {
            size_t id = G.alpha.idx(i, j, k_mid);
            file << i << "," << j << "," << G.alpha.ptr()[id] << "\n";
        }
    }
    file.close();
    printf("Lapse slice exported to %s\n", filename.c_str());
}

template <typename T>
void export_grid_structure(const BSSNGridSoA<T> &G, const std::string &filename) {
    const auto &st = G.st;
    const auto &D = G.dims;

    std::ofstream file(filename);
    file << "i,j,chi,zone_type\n";

    size_t k_mid = st.nz_tot / 2;

    for (size_t i = 0; i < st.nx_tot; ++i) {
        for (size_t j = 0; j < st.ny_tot; ++j) {
            size_t id = i * st.sx + j * st.sy + k_mid;

            int zone = (i >= D.ng && i < D.ng + D.nx && j >= D.ng && j < D.ng + D.ny) ? 1 : 0;

            file << i << "," << j << "," << G.chi.ptr()[id] << "," << zone << "\n";
        }
    }
    file.close();
    printf("Full grid structure exported to %s\n", filename.c_str());
}
static void test_halo_periodic_alpha() {
    printf("=== Halo periodic test (alpha) ===\n");

    constexpr size_t    nx = 8, ny = 6, nz = 10, ng = 2;
    BSSNGridSoA<double> G(nx, ny, nz, ng, 1.0, 1.0, 1.0);

    fill_interior(G.alpha, G);
    apply_halos_grid<BoundaryPeriodic>(G);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    printf("[CHECK] X halos...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t iL = I0 - 1 - g;
        const size_t iR = I1 + g;
        const size_t isrcL = I1 - 1 - g;
        const size_t isrcR = I0 + g;

        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const double aL = G.alpha.ptr()[G.alpha.idx(iL, j, k)];
                const double eL = G.alpha.ptr()[G.alpha.idx(isrcL, j, k)];
                const double aR = G.alpha.ptr()[G.alpha.idx(iR, j, k)];
                const double eR = G.alpha.ptr()[G.alpha.idx(isrcR, j, k)];
                assert(aL == eL);
                assert(aR == eR);
            }
    }
    printf("  OK\n");

    printf("[CHECK] Y halos...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t jL = J0 - 1 - g;
        const size_t jR = J1 + g;
        const size_t jsrcL = J1 - 1 - g;
        const size_t jsrcR = J0 + g;

        for (size_t i = I0; i < I1; ++i)
            for (size_t k = K0; k < K1; ++k) {
                const double aL = G.alpha.ptr()[G.alpha.idx(i, jL, k)];
                const double eL = G.alpha.ptr()[G.alpha.idx(i, jsrcL, k)];
                const double aR = G.alpha.ptr()[G.alpha.idx(i, jR, k)];
                const double eR = G.alpha.ptr()[G.alpha.idx(i, jsrcR, k)];
                assert(aL == eL);
                assert(aR == eR);
            }
    }
    printf("  OK\n");

    printf("[CHECK] Z halos...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t kL = K0 - 1 - g;
        const size_t kR = K1 + g;
        const size_t ksrcL = K1 - 1 - g;
        const size_t ksrcR = K0 + g;

        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j) {
                const double aL = G.alpha.ptr()[G.alpha.idx(i, j, kL)];
                const double eL = G.alpha.ptr()[G.alpha.idx(i, j, ksrcL)];
                const double aR = G.alpha.ptr()[G.alpha.idx(i, j, kR)];
                const double eR = G.alpha.ptr()[G.alpha.idx(i, j, ksrcR)];
                assert(aL == eL);
                assert(aR == eR);
            }
    }
    printf("  OK\n");

    printf("=== Halo periodic test PASSED ===\n\n");
}

static void test_halo_clamp_alpha() {
    printf("=== Halo clamp test (alpha) ===\n");

    constexpr size_t    nx = 8, ny = 6, nz = 10, ng = 2;
    BSSNGridSoA<double> G(nx, ny, nz, ng, 1.0, 1.0, 1.0);

    fill_interior(G.alpha, G);
    apply_halos_grid<BoundaryClamp>(G);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    printf("[CHECK] X halos clamp to nearest interior...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t iL = I0 - 1 - g;
        const size_t iR = I1 + g;

        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                assert(G.alpha.ptr()[G.alpha.idx(iL, j, k)] ==
                       G.alpha.ptr()[G.alpha.idx(I0, j, k)]);
                assert(G.alpha.ptr()[G.alpha.idx(iR, j, k)] ==
                       G.alpha.ptr()[G.alpha.idx(I1 - 1, j, k)]);
            }
    }
    printf("  OK\n");

    printf("[CHECK] Y halos clamp to nearest interior...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t jL = J0 - 1 - g;
        const size_t jR = J1 + g;

        for (size_t i = I0; i < I1; ++i)
            for (size_t k = K0; k < K1; ++k) {
                assert(G.alpha.ptr()[G.alpha.idx(i, jL, k)] ==
                       G.alpha.ptr()[G.alpha.idx(i, J0, k)]);
                assert(G.alpha.ptr()[G.alpha.idx(i, jR, k)] ==
                       G.alpha.ptr()[G.alpha.idx(i, J1 - 1, k)]);
            }
    }
    printf("  OK\n");

    printf("[CHECK] Z halos clamp to nearest interior...\n");
    for (size_t g = 0; g < ng; ++g) {
        const size_t kL = K0 - 1 - g;
        const size_t kR = K1 + g;

        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j) {
                assert(G.alpha.ptr()[G.alpha.idx(i, j, kL)] ==
                       G.alpha.ptr()[G.alpha.idx(i, j, K0)]);
                assert(G.alpha.ptr()[G.alpha.idx(i, j, kR)] ==
                       G.alpha.ptr()[G.alpha.idx(i, j, K1 - 1)]);
            }
    }
    printf("  OK\n");

    printf("=== Halo clamp test PASSED ===\n\n");
}
#include <vector>

namespace {

template <typename T> struct ConvMetrics {
    double Einf = 0.0;
    double EL2 = 0.0;
    size_t n = 0;
};

template <typename T>
static inline double scalar_R_at(const tensorium_RG::BSSNGridSoA<T> &G,
                                 const tensorium_RG::Field3D<T> *Ricci6, size_t id) {
    const double chi = (double)G.chi.ptr()[id];

    const double gixx = (double)G.gamma_tilde_inv[tensorium_RG::XX].ptr()[id];
    const double gixy = (double)G.gamma_tilde_inv[tensorium_RG::XY].ptr()[id];
    const double gixz = (double)G.gamma_tilde_inv[tensorium_RG::XZ].ptr()[id];
    const double giyy = (double)G.gamma_tilde_inv[tensorium_RG::YY].ptr()[id];
    const double giyz = (double)G.gamma_tilde_inv[tensorium_RG::YZ].ptr()[id];
    const double gizz = (double)G.gamma_tilde_inv[tensorium_RG::ZZ].ptr()[id];

    const double Rxx = (double)Ricci6[tensorium_RG::XX].ptr()[id];
    const double Rxy = (double)Ricci6[tensorium_RG::XY].ptr()[id];
    const double Rxz = (double)Ricci6[tensorium_RG::XZ].ptr()[id];
    const double Ryy = (double)Ricci6[tensorium_RG::YY].ptr()[id];
    const double Ryz = (double)Ricci6[tensorium_RG::YZ].ptr()[id];
    const double Rzz = (double)Ricci6[tensorium_RG::ZZ].ptr()[id];

    const double tr_conf =
        gixx * Rxx + giyy * Ryy + gizz * Rzz + 2.0 * (gixy * Rxy + gixz * Rxz + giyz * Ryz);

    return chi * tr_conf;
}

template <typename T>
ConvMetrics<T> measure_R_error(const tensorium_RG::BSSNGridSoA<T> &G,
                               const tensorium_RG::Field3D<T> *Ricci6, T M, double r_cut_phys) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 2, i1 = I1 - 2;
    const size_t j0 = J0 + 2, j1 = J1 - 2;
    const size_t k0 = K0 + 2, k1 = K1 - 2;

    double sum2 = 0.0;
    double maxa = 0.0;
    size_t n = 0;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.gamma_tilde[tensorium_RG::XX].idx(i, j, k);

                T x, y, z;
                G.coords(i, j, k, x, y, z);
                const double r = std::sqrt((double)x * (double)x + (double)y * (double)y +
                                           (double)z * (double)z);
                if (r <= r_cut_phys)
                    continue;

                const double R = scalar_R_at(G, Ricci6, id);
                const double a = std::abs(R);

                if (a > maxa)
                    maxa = a;
                sum2 += R * R;
                ++n;
            }

    ConvMetrics<T> m;
    m.Einf = maxa;
    m.n = n;
    m.EL2 = (n > 0) ? std::sqrt(sum2 / (double)n) : 0.0;
    return m;
}

static inline double order_from(double E1, double E2) {
    if (E2 <= 0.0 || E1 <= 0.0)
        return 0.0;
    return std::log(E1 / E2) / std::log(2.0);
}

} // namespace
void test_convergence_schwarzschild_R() {
    using T = double;

    const T      M = 1.0;
    const int    ng = 3;  
    const double L = 5.0 * M; 
    const double r_cut = 2.0 * M;

    std::vector<int> Ns = {32, 64, 128}; 

    std::vector<double> Einf, EL2, hs;

    for (int N : Ns) {
        const int    nx = N, ny = N, nz = N;
        const double dx = (2.0 * L) / (double)(nx - 1);
        const double dy = dx;
        const double dz = dx;

        tensorium_RG::BSSNGridSoA<T> G(nx, ny, nz, ng, (T)dx, (T)dy, (T)dz);
        G.x0 = (T)(-L);
        G.y0 = (T)(-L);
        G.z0 = (T)(-L);

        tensorium_RG::Field3D<T> Ricci6[6];
        for (int s = 0; s < 6; ++s)
            Ricci6[s] = tensorium_RG::make_field<T>(G.st);

        tensorium_RG::init::schwarzschild_isotropic(G, M, 0.0, 0.0, 0.0, 1e-6);

        tensorium_RG::bssn::compute_ricci_bssn(G, Ricci6);
        tensorium_RG::apply_halos_grid<tensorium_RG::BoundaryClamp>(G);
        for (int s = 0; s < 6; ++s)
            tensorium_RG::BoundaryClamp::apply(Ricci6[s], G.dims);

        auto m = measure_R_error(G, Ricci6, M, r_cut);

        hs.push_back(dx);
        Einf.push_back(m.Einf);
        EL2.push_back(m.EL2);

        std::printf("[CONV] N=%d dx=%.6e  Einf=%.6e  EL2=%.6e  samples=%zu\n", N, dx, m.Einf, m.EL2,
                    m.n);
    }

    if (Ns.size() >= 2) {
        const double p_inf = order_from(Einf[0], Einf[1]);
        const double p_l2 = order_from(EL2[0], EL2[1]);
        std::printf("[CONV] p_inf=%.3f  p_L2=%.3f (expect ~4 if truly 4th order)\n", p_inf, p_l2);
    }
}

int grid_tests() {
    printf("=== Grid basic tests ===\n");
    const size_t nx = 256;
    const size_t ny = 256;
    const size_t nz = 256;
    const size_t ng = 3;

    const double dx = 0.05;
    const double dy = 0.05;
    const double dz = 0.05;

    printf("Allocating grid %zux%zux%zu (ng=%zu)\n", nx, ny, nz, ng);

    BSSNGridSoA<double> G(nx, ny, nz, ng, dx, dy, dz);
    print_dims(G.dims);
    print_strides(G.st);

    printf("[TEST] Stride invariants...\n");
    assert(G.st.sz == 1);
    assert(G.st.sy == G.st.nz_tot);
    assert(G.st.sx == G.st.ny_tot * G.st.nz_tot);
    printf("  OK\n");

    printf("[TEST] Alignment...\n");
    uintptr_t p = reinterpret_cast<uintptr_t>(G.alpha.ptr());
    printf("  alpha ptr = %p\n", (void *)p);
    assert(p % TENSORIUM_ALIGN == 0);
    printf("  OK (alignment = %d bytes)\n", TENSORIUM_ALIGN);

    printf("[TEST] Indexing round-trip...\n");
    size_t i = G.dims.ng;
    size_t j = G.dims.ng;
    size_t k = G.dims.ng;

    size_t idx = G.alpha.idx(i, j, k);
    printf("  idx(%zu,%zu,%zu) = %zu\n", i, j, k, idx);

    G.alpha.ptr()[idx] = 42.0;
    assert(G.alpha.ptr()[idx] == 42.0);
    printf("  OK (value = %.1f)\n", G.alpha.ptr()[idx]);

    printf("[TEST] Linear layout sanity (k contiguous)...\n");
    size_t idx0 = G.alpha.idx(i, j, k);
    size_t idx1 = G.alpha.idx(i, j, k + 1);
    printf("  idx(k)   = %zu\n", idx0);
    printf("  idx(k+1) = %zu\n", idx1);
    assert(idx1 == idx0 + 1);
    printf("  OK\n");

    printf("[TEST] Distinct fields have distinct storage...\n");
    G.alpha.ptr()[idx] = 1.0;
    G.chi.ptr()[idx] = 2.0;
    assert(G.alpha.ptr()[idx] == 1.0);
    assert(G.chi.ptr()[idx] == 2.0);
    printf("  OK\n");

    printf("[TEST] Coordinate mapping...\n");
    double x, y, z;
    G.coords(i, j, k, x, y, z);
    printf("  coords = (%g, %g, %g)\n", x, y, z);
    assert(x == 0.0 && y == 0.0 && z == 0.0);
    printf("  OK\n");

    test_halo_periodic_alpha();
    test_halo_clamp_alpha();
    printf("=== Grid basic tests PASSED ===\n\n");
    BSSNGridSoA<double> G2(256, 256, 256, 3, 0.05, 0.05, 0.05);
    G2.x0 = -(double(G2.dims.nx) * G2.dx) / 2.0;
    G2.y0 = -(double(G2.dims.ny) * G2.dy) / 2.0;
    G2.z0 = -(double(G2.dims.nz) * G2.dz) / 2.0;
    init::schwarzschild_isotropic(G2, /*M=*/1.0, /*center=*/0.0, 0.0, 0.0, /*r_floor=*/1e-6);
    apply_halos_grid<BoundaryClamp>(G2);
    test_convergence_schwarzschild_R();
    export_chi_slice(G2, "chi_slice.csv");
    export_grid_structure(G2, "grid_structure.csv");
    export_alpha_slice(G2, "alpha_slice.csv");
    export_log_chi_slice(G2, "log_chi_slice.csv");
	init::minkowski(G2, /*M=*/1.0, /*center=*/0.0, 0.0, 0.0, /*r_floor=*/1e-6);
    apply_halos_grid<BoundaryClamp>(G2);

    return 0;
}
