#pragma once

#if defined(TENSORIUM_USE_MPI) && !defined(TENSORIUM_ENABLE_MPI)
#    define TENSORIUM_ENABLE_MPI
#endif

/**
 * @file MPIDomain.hpp
 * @brief 3D Cartesian domain decomposition for distributed BSSN grids.
 * @details
 * Creates an MPI Cartesian topology and computes local grid dimensions,
 * neighbor ranks, and coordinate mappings. Supports 1D, 2D, or 3D decomposition.
 *
 * The decomposition divides the global grid [Nx, Ny, Nz] across [px, py, pz] processes.
 * Each process owns a local subdomain plus ghost zones for halo exchange.
 */

#ifdef TENSORIUM_ENABLE_MPI
#    include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace tensorium::mpi {

/**
 * @brief Neighbor direction indices for 3D Cartesian topology.
 */
enum NeighborDir : int {
    X_MINUS = 0,
    X_PLUS = 1,
    Y_MINUS = 2,
    Y_PLUS = 3,
    Z_MINUS = 4,
    Z_PLUS = 5,
    NUM_NEIGHBORS = 6
};

/**
 * @brief Configuration for domain decomposition.
 */
struct DomainConfig {
    size_t global_nx = 0;
    size_t global_ny = 0;
    size_t global_nz = 0;

    size_t ng = 6;

    int proc_x = 0;
    int proc_y = 0;
    int proc_z = 0;

    // Physical domain extent
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;
    double z0 = 0.0, z1 = 1.0;

    // Periodicity (typically false for BSSN with Sommerfeld BC)
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;
};

#ifdef TENSORIUM_ENABLE_MPI

/**
 * @brief 3D Cartesian domain decomposition manager.
 */
class MPIDomain {
  public:
    /**
     * @brief Create domain decomposition from configuration.
     * @param config Domain configuration with global sizes and process grid.
     * @param parent Parent communicator (default: MPI_COMM_WORLD).
     */
    explicit MPIDomain(const DomainConfig &config, MPI_Comm parent = MPI_COMM_WORLD)
        : config_(config),
          parent_comm_(parent) {

        int world_size, world_rank;
        MPI_Comm_size(parent, &world_size);
        MPI_Comm_rank(parent, &world_rank);

        // Determine process grid dimensions
        dims_[0] = config.proc_x;
        dims_[1] = config.proc_y;
        dims_[2] = config.proc_z;

        if (dims_[0] == 0 || dims_[1] == 0 || dims_[2] == 0) {
            int total_procs = world_size;
            MPI_Dims_create(total_procs, 3, dims_.data());
        }

        int expected_procs = dims_[0] * dims_[1] * dims_[2];
        if (expected_procs != world_size) {
            throw std::runtime_error("Process grid " + std::to_string(dims_[0]) + "x" +
                                     std::to_string(dims_[1]) + "x" + std::to_string(dims_[2]) +
                                     " = " + std::to_string(expected_procs) +
                                     " doesn't match world size " + std::to_string(world_size));
        }

        periods_[0] = config.periodic_x ? 1 : 0;
        periods_[1] = config.periodic_y ? 1 : 0;
        periods_[2] = config.periodic_z ? 1 : 0;

        MPI_Cart_create(parent, 3, dims_.data(), periods_.data(), 1, &cart_comm_);
        MPI_Comm_rank(cart_comm_, &cart_rank_);
        MPI_Cart_coords(cart_comm_, cart_rank_, 3, coords_.data());

        MPI_Cart_shift(cart_comm_, 0, 1, &neighbors_[X_MINUS], &neighbors_[X_PLUS]);
        MPI_Cart_shift(cart_comm_, 1, 1, &neighbors_[Y_MINUS], &neighbors_[Y_PLUS]);
        MPI_Cart_shift(cart_comm_, 2, 1, &neighbors_[Z_MINUS], &neighbors_[Z_PLUS]);

        compute_local_extents();
    }

    ~MPIDomain() {
        if (cart_comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_comm_);
        }
    }

    MPIDomain(const MPIDomain &) = delete;
    MPIDomain &operator=(const MPIDomain &) = delete;

    MPIDomain(MPIDomain &&other) noexcept { swap(other); }
    MPIDomain &operator=(MPIDomain &&other) noexcept {
        if (this != &other) {
            if (cart_comm_ != MPI_COMM_NULL) {
                MPI_Comm_free(&cart_comm_);
            }
            swap(other);
        }
        return *this;
    }

    /// @brief Get the Cartesian communicator.
    MPI_Comm comm() const noexcept { return cart_comm_; }

    /// @brief Get rank in Cartesian communicator.
    int rank() const noexcept { return cart_rank_; }

    /// @brief Get process coordinates in the Cartesian grid.
    const std::array<int, 3> &coords() const noexcept { return coords_; }

    /// @brief Get process grid dimensions.
    const std::array<int, 3> &dims() const noexcept { return dims_; }

    /// @brief Get neighbor rank in specified direction (MPI_PROC_NULL if none).
    int neighbor(NeighborDir dir) const noexcept { return neighbors_[dir]; }

    /// @brief Check if this process has a neighbor in the given direction.
    bool has_neighbor(NeighborDir dir) const noexcept { return neighbors_[dir] != MPI_PROC_NULL; }

    /// @brief Check if this process is at a physical boundary.
    bool is_boundary_x_minus() const noexcept { return !has_neighbor(X_MINUS); }
    bool is_boundary_x_plus() const noexcept { return !has_neighbor(X_PLUS); }
    bool is_boundary_y_minus() const noexcept { return !has_neighbor(Y_MINUS); }
    bool is_boundary_y_plus() const noexcept { return !has_neighbor(Y_PLUS); }
    bool is_boundary_z_minus() const noexcept { return !has_neighbor(Z_MINUS); }
    bool is_boundary_z_plus() const noexcept { return !has_neighbor(Z_PLUS); }

    // ========== Local grid information ==========

    /// @brief Local number of physical cells in X (excluding ghosts).
    size_t local_nx() const noexcept { return local_nx_; }
    size_t local_ny() const noexcept { return local_ny_; }
    size_t local_nz() const noexcept { return local_nz_; }

    /// @brief Number of ghost cells.
    size_t ng() const noexcept { return config_.ng; }

    /// @brief Global starting index for this process (in global coordinates).
    size_t global_i0() const noexcept { return global_i0_; }
    size_t global_j0() const noexcept { return global_j0_; }
    size_t global_k0() const noexcept { return global_k0_; }

    /// @brief Physical coordinate origin for this subdomain.
    double local_x0() const noexcept { return local_x0_; }
    double local_y0() const noexcept { return local_y0_; }
    double local_z0() const noexcept { return local_z0_; }

    /// @brief Grid spacing (uniform, same as global).
    double dx() const noexcept { return dx_; }
    double dy() const noexcept { return dy_; }
    double dz() const noexcept { return dz_; }

    /// @brief Global grid dimensions.
    size_t global_nx() const noexcept { return config_.global_nx; }
    size_t global_ny() const noexcept { return config_.global_ny; }
    size_t global_nz() const noexcept { return config_.global_nz; }

    /// @brief Check if this is the root process (rank 0 in Cartesian comm).
    bool is_root() const noexcept { return cart_rank_ == 0; }

    /// @brief Barrier on Cartesian communicator.
    void barrier() const { MPI_Barrier(cart_comm_); }

    /// @brief Print domain decomposition info (from root only).
    void print_info() const {
        if (is_root()) {
            printf("=== MPI Domain Decomposition ===\n");
            printf("  Global grid: %zu x %zu x %zu\n", config_.global_nx, config_.global_ny,
                   config_.global_nz);
            printf("  Process grid: %d x %d x %d = %d processes\n", dims_[0], dims_[1], dims_[2],
                   dims_[0] * dims_[1] * dims_[2]);
            printf("  Ghost cells: %zu\n", config_.ng);
            printf("  Periodic: [%s, %s, %s]\n", periods_[0] ? "yes" : "no",
                   periods_[1] ? "yes" : "no", periods_[2] ? "yes" : "no");
            printf("================================\n");
            fflush(stdout);
        }
        barrier();

        // Each process prints its local info
        for (int r = 0; r < dims_[0] * dims_[1] * dims_[2]; ++r) {
            if (cart_rank_ == r) {
                printf("  Rank %d: coords=(%d,%d,%d), local=(%zu,%zu,%zu), "
                       "global_start=(%zu,%zu,%zu), x0=(%.3f,%.3f,%.3f)\n",
                       cart_rank_, coords_[0], coords_[1], coords_[2], local_nx_, local_ny_,
                       local_nz_, global_i0_, global_j0_, global_k0_, local_x0_, local_y0_,
                       local_z0_);
                fflush(stdout);
            }
            barrier();
        }
    }

  private:
    void swap(MPIDomain &other) noexcept {
        std::swap(config_, other.config_);
        std::swap(parent_comm_, other.parent_comm_);
        std::swap(cart_comm_, other.cart_comm_);
        std::swap(cart_rank_, other.cart_rank_);
        std::swap(dims_, other.dims_);
        std::swap(periods_, other.periods_);
        std::swap(coords_, other.coords_);
        std::swap(neighbors_, other.neighbors_);
        std::swap(local_nx_, other.local_nx_);
        std::swap(local_ny_, other.local_ny_);
        std::swap(local_nz_, other.local_nz_);
        std::swap(global_i0_, other.global_i0_);
        std::swap(global_j0_, other.global_j0_);
        std::swap(global_k0_, other.global_k0_);
        std::swap(local_x0_, other.local_x0_);
        std::swap(local_y0_, other.local_y0_);
        std::swap(local_z0_, other.local_z0_);
        std::swap(dx_, other.dx_);
        std::swap(dy_, other.dy_);
        std::swap(dz_, other.dz_);
    }

    void compute_local_extents() {
        dx_ = (config_.x1 - config_.x0) / static_cast<double>(config_.global_nx);
        dy_ = (config_.y1 - config_.y0) / static_cast<double>(config_.global_ny);
        dz_ = (config_.z1 - config_.z0) / static_cast<double>(config_.global_nz);

        auto divide_grid = [](size_t global_n, int nprocs, int coord) -> std::pair<size_t, size_t> {
            size_t base = global_n / static_cast<size_t>(nprocs);
            size_t remainder = global_n % static_cast<size_t>(nprocs);

            size_t local_n = base + (static_cast<size_t>(coord) < remainder ? 1 : 0);

            size_t start =
                base * static_cast<size_t>(coord) + std::min(static_cast<size_t>(coord), remainder);

            return {local_n, start};
        };

        auto [nx, i0] = divide_grid(config_.global_nx, dims_[0], coords_[0]);
        auto [ny, j0] = divide_grid(config_.global_ny, dims_[1], coords_[1]);
        auto [nz, k0] = divide_grid(config_.global_nz, dims_[2], coords_[2]);

        local_nx_ = nx;
        local_ny_ = ny;
        local_nz_ = nz;
        global_i0_ = i0;
        global_j0_ = j0;
        global_k0_ = k0;

        local_x0_ = config_.x0 + static_cast<double>(global_i0_) * dx_;
        local_y0_ = config_.y0 + static_cast<double>(global_j0_) * dy_;
        local_z0_ = config_.z0 + static_cast<double>(global_k0_) * dz_;
    }

    DomainConfig config_;
    MPI_Comm     parent_comm_ = MPI_COMM_WORLD;
    MPI_Comm     cart_comm_ = MPI_COMM_NULL;
    int          cart_rank_ = 0;

    std::array<int, 3>             dims_ = {0, 0, 0};
    std::array<int, 3>             periods_ = {0, 0, 0};
    std::array<int, 3>             coords_ = {0, 0, 0};
    std::array<int, NUM_NEIGHBORS> neighbors_ = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                                                 MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t local_nx_ = 0, local_ny_ = 0, local_nz_ = 0;
    size_t global_i0_ = 0, global_j0_ = 0, global_k0_ = 0;
    double local_x0_ = 0, local_y0_ = 0, local_z0_ = 0;
    double dx_ = 0, dy_ = 0, dz_ = 0;
};

#else // !TENSORIUM_ENABLE_MPI

/**
 * @brief Stub implementation when MPI is disabled (single-process).
 */
class MPIDomain {
  public:
    explicit MPIDomain(const DomainConfig &config)
        : config_(config),
          local_nx_(config.global_nx),
          local_ny_(config.global_ny),
          local_nz_(config.global_nz) {
        dx_ = (config.x1 - config.x0) / static_cast<double>(config.global_nx);
        dy_ = (config.y1 - config.y0) / static_cast<double>(config.global_ny);
        dz_ = (config.z1 - config.z0) / static_cast<double>(config.global_nz);
        local_x0_ = config.x0;
        local_y0_ = config.y0;
        local_z0_ = config.z0;
    }

    int                       rank() const noexcept { return 0; }
    const std::array<int, 3> &coords() const noexcept { return coords_; }
    const std::array<int, 3> &dims() const noexcept { return dims_; }
    int                       neighbor(NeighborDir) const noexcept { return -1; }
    bool                      has_neighbor(NeighborDir) const noexcept { return false; }

    bool is_boundary_x_minus() const noexcept { return true; }
    bool is_boundary_x_plus() const noexcept { return true; }
    bool is_boundary_y_minus() const noexcept { return true; }
    bool is_boundary_y_plus() const noexcept { return true; }
    bool is_boundary_z_minus() const noexcept { return true; }
    bool is_boundary_z_plus() const noexcept { return true; }

    size_t local_nx() const noexcept { return local_nx_; }
    size_t local_ny() const noexcept { return local_ny_; }
    size_t local_nz() const noexcept { return local_nz_; }
    size_t ng() const noexcept { return config_.ng; }

    size_t global_i0() const noexcept { return 0; }
    size_t global_j0() const noexcept { return 0; }
    size_t global_k0() const noexcept { return 0; }

    double local_x0() const noexcept { return local_x0_; }
    double local_y0() const noexcept { return local_y0_; }
    double local_z0() const noexcept { return local_z0_; }
    double dx() const noexcept { return dx_; }
    double dy() const noexcept { return dy_; }
    double dz() const noexcept { return dz_; }

    size_t global_nx() const noexcept { return config_.global_nx; }
    size_t global_ny() const noexcept { return config_.global_ny; }
    size_t global_nz() const noexcept { return config_.global_nz; }

    bool is_root() const noexcept { return true; }
    void barrier() const {}
    void print_info() const {
        printf("=== Single-Process Domain (MPI Disabled) ===\n");
        printf("  Grid: %zu x %zu x %zu\n", local_nx_, local_ny_, local_nz_);
        printf("=============================================\n");
    }

  private:
    DomainConfig       config_;
    std::array<int, 3> dims_ = {1, 1, 1};
    std::array<int, 3> coords_ = {0, 0, 0};
    size_t             local_nx_, local_ny_, local_nz_;
    double             local_x0_, local_y0_, local_z0_;
    double             dx_, dy_, dz_;
};

#endif // TENSORIUM_ENABLE_MPI

} // namespace tensorium::mpi
