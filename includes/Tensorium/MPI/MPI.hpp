#pragma once

/**
 * @file MPI.hpp
 * @brief Master include for Tensorium MPI support.
 * @details
 * Include this single header to get access to all MPI functionality:
 * - MPIContext: RAII MPI initialization
 * - MPIDomain: 3D Cartesian domain decomposition
 * - HaloExchanger: Non-blocking halo exchange
 * - Reductions: Global reduction operations
 * - MPIBSSNIntegration: BSSN grid creation and boundary handling
 *
 * @note Define TENSORIUM_ENABLE_MPI before including this header to enable
 * MPI functionality. Without this define, stub implementations are provided
 * that allow single-process execution.
 *
 * Example usage:
 * @code
 * #define TENSORIUM_ENABLE_MPI
 * #include "Tensorium/MPI/MPI.hpp"
 *
 * int main(int argc, char** argv) {
 *     using namespace tensorium::mpi;
 *
 *     // Initialize MPI
 *     MPIContext ctx(argc, argv);
 *
 *     // Configure domain
 *     auto config = make_domain_config(128, 128, 128, 6, 256.0);
 *
 *     // Create domain decomposition
 *     MPIDomain domain(config);
 *     domain.print_info();
 *
 *     // Create local BSSN grid
 *     auto grid = create_local_bssn_grid<double>(domain);
 *
 *     // Create MPI boundary handler
 *     MPIBoundary<double> boundary(domain, grid->alpha.st,
 *                                   domain.local_nx(), domain.local_ny(), domain.local_nz());
 *
 *     // ... setup initial data ...
 *
 *     // Evolution loop
 *     Reductions red(domain);
 *     for (int step = 0; step < nsteps; ++step) {
 *         // Compute global CFL time step
 *         double dt = compute_global_dt_cfl(*grid, domain, 0.25);
 *
 *         // Exchange halos before RHS evaluation
 *         boundary.exchange_all_halos(*grid);
 *
 *         // ... RK4 sub-steps with halo exchanges ...
 *     }
 *
 *     return 0;
 * }
 * @endcode
 */

#include "MPIBSSNIntegration.hpp"
#include "MPIContext.hpp"
#include "MPIDomain.hpp"
#include "MPIHaloExchange.hpp"
#include "MPIReductions.hpp"
