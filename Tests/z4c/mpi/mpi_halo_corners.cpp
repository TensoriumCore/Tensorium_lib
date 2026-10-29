#include "Tensorium/MPI/MPIHaloExchange.hpp"
#include "Tensorium/MPI/MPIBSSNRK4.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <limits>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int result = 0;
    {
        tensorium::mpi::DomainConfig config;
        config.global_nx = 16;
        config.global_ny = 16;
        config.global_nz = 8;
        config.ng = 4;
        config.proc_x = 2;
        config.proc_y = 2;
        config.proc_z = 1;
        config.periodic_x = true;
        config.periodic_y = true;
        tensorium::mpi::MPIDomain domain(config);
        tensorium_RG::BSSNGridSoA<double> grid(domain.local_nx(), domain.local_ny(),
                                                domain.local_nz(), domain.ng(), 1.0, 1.0, 1.0);
        auto &field = grid.alpha;
        const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
        std::fill_n(field.ptr(), total, std::numeric_limits<double>::quiet_NaN());
        const auto value = [](size_t x, size_t y, size_t z) {
            return double(x + 100 * y + 10000 * z);
        };
        for (size_t i = 0; i < domain.local_nx(); ++i)
            for (size_t j = 0; j < domain.local_ny(); ++j)
                for (size_t k = 0; k < domain.local_nz(); ++k)
                    field.ptr()[field.idx(i + config.ng, j + config.ng, k + config.ng)] =
                        value(domain.global_i0() + i, domain.global_j0() + j, k);

        tensorium::mpi::HaloExchanger<double> exchanger(domain, field.st, domain.local_nx(),
                                                        domain.local_ny(), domain.local_nz());
        exchanger.exchange_blocking(field.ptr());
        int failures = 0;
        for (size_t i = 0; i < domain.local_nx() + 2 * config.ng; ++i)
            for (size_t j = 0; j < domain.local_ny() + 2 * config.ng; ++j)
                for (size_t k = config.ng; k < config.ng + domain.local_nz(); ++k) {
                    const size_t x = (domain.global_i0() + i + config.global_nx - config.ng) %
                                     config.global_nx;
                    const size_t y = (domain.global_j0() + j + config.global_ny - config.ng) %
                                     config.global_ny;
                    if (field.ptr()[field.idx(i, j, k)] != value(x, y, k - config.ng))
                        ++failures;
                }
        int global_failures = 0;
        MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, domain.comm());
        if (domain.is_root())
            std::printf("MPI halo face and corner errors: %d\n", global_failures);
        result = global_failures == 0 ? 0 : 1;

        // Ricci must be rebuilt on every physical cell, including MPI interfaces.
        for (size_t id = 0; id < total; ++id) {
            grid.alpha.ptr()[id] = grid.chi.ptr()[id] = 1.0;
            grid.K.ptr()[id] = grid.Theta.ptr()[id] = 0.0;
            for (int c = 0; c < 3; ++c)
                grid.beta[c].ptr()[id] = grid.B[c].ptr()[id] =
                    grid.tildeGamma[c].ptr()[id] = grid.Z[c].ptr()[id] = 0.0;
            for (int s = 0; s < 6; ++s) {
                const double diag = (s == tensorium_RG::XX || s == tensorium_RG::YY ||
                                     s == tensorium_RG::ZZ) ? 1.0 : 0.0;
                grid.gamma_tilde[s].ptr()[id] = grid.gamma_tilde_inv[s].ptr()[id] = diag;
                grid.A_tilde[s].ptr()[id] = 0.0;
                grid.Ricci[s].ptr()[id] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        tensorium::mpi::MPIBSSNRKStepper<double> stepper(domain, grid, 0);
        stepper.set_state_log_stride(10000);
        stepper.step(grid, 0.01, 0);
        int ricci_failures = 0;
        for (size_t i = config.ng; i < config.ng + domain.local_nx(); ++i)
            for (size_t j = config.ng; j < config.ng + domain.local_ny(); ++j)
                for (size_t k = config.ng; k < config.ng + domain.local_nz(); ++k)
                    for (int s = 0; s < 6; ++s)
                        if (!std::isfinite(grid.Ricci[s].ptr()[grid.alpha.idx(i, j, k)]))
                            ++ricci_failures;
        MPI_Allreduce(&ricci_failures, &global_failures, 1, MPI_INT, MPI_SUM, domain.comm());
        if (domain.is_root())
            std::printf("MPI physical Ricci cells not rebuilt: %d\n", global_failures);
        if (global_failures != 0)
            result = 1;
    }
    {
        // A physical x face intersects a y-process interface. Its extrapolated x ghosts
        // must be sent to the adjacent y rank before evaluating mixed derivatives.
        tensorium::mpi::DomainConfig config;
        config.global_nx = config.global_ny = 16;
        config.global_nz = 8;
        config.ng = 4;
        config.proc_x = config.proc_y = 2;
        config.proc_z = 1;
        tensorium::mpi::MPIDomain domain(config);
        tensorium_RG::BSSNGridSoA<double> grid(domain.local_nx(), domain.local_ny(),
                                                domain.local_nz(), domain.ng(), 1.0, 1.0, 1.0);
        auto &field = grid.alpha;
        const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
        std::fill_n(field.ptr(), total, std::numeric_limits<double>::quiet_NaN());
        for (size_t i = 0; i < domain.local_nx(); ++i)
            for (size_t j = 0; j < domain.local_ny(); ++j)
                for (size_t k = 0; k < domain.local_nz(); ++k)
                    field.ptr()[field.idx(i + config.ng, j + config.ng, k + config.ng)] =
                        1.0 + 0.02 * double(domain.global_i0() + i) +
                        0.03 * double(domain.global_j0() + j);
        tensorium::mpi::MPIBoundary<double> boundary(domain, field.st, domain.local_nx(),
                                                      domain.local_ny(), domain.local_nz());
        boundary.exchange_halos(grid, tensorium::mpi::HaloFieldAlpha);
        const size_t i = domain.coords()[0] == 0 ? config.ng - 1 : config.ng + domain.local_nx();
        const size_t j = domain.coords()[1] == 0 ? config.ng + domain.local_ny() : config.ng - 1;
        const long x = long(domain.global_i0()) + long(i) - long(config.ng);
        const long y = long(domain.global_j0()) + long(j) - long(config.ng);
        const double expected = 1.0 + 0.02 * double(x) + 0.03 * double(y);
        const double actual = field.ptr()[field.idx(i, j, config.ng + 2)];
        int local_failure = !std::isfinite(actual) || std::abs(actual - expected) > 1e-12;
        int global_failures = 0;
        MPI_Allreduce(&local_failure, &global_failures, 1, MPI_INT, MPI_SUM, domain.comm());
        if (domain.is_root())
            std::printf("MPI/physical boundary corner errors: %d\n", global_failures);
        if (global_failures != 0)
            result = 1;
    }
    MPI_Finalize();
    return result;
}
