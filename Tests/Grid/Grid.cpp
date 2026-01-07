#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/BSSNInitialData.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/BSSNMetricSplit.hpp"
#include "../includes/Tensorium_Grid/GridSetup.hpp"
#include "../test.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>

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
int grid_tests() {
    printf("=== Grid basic tests ===\n");
    const size_t nx = 128;
    const size_t ny = 128;
    const size_t nz = 128;
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

    init::schwarzschild_isotropic(G2, /*M=*/1.0, /*center=*/0.0, 0.0, 0.0, /*r_floor=*/1e-6);
    apply_halos_grid<BoundaryClamp>(G2); 
    return 0;
}
