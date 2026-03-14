#pragma once

#if defined(TENSORIUM_USE_MPI) && !defined(TENSORIUM_ENABLE_MPI)
#    define TENSORIUM_ENABLE_MPI
#endif

/**
 * @file MPIHaloExchange.hpp
 * @brief Non-blocking MPI halo exchange for 3D Field arrays.
 * @details
 * Implements efficient halo exchange using MPI derived datatypes for face regions.
 * Supports both blocking and non-blocking communication patterns for overlap
 * with computation.
 *
 * The halo exchanger creates MPI_Datatype for each face of the local subdomain,
 * enabling direct send/recv of strided data without manual packing.
 */

#ifdef TENSORIUM_ENABLE_MPI
#    include <mpi.h>
#endif

#include "MPIDomain.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace tensorium::mpi {

#ifdef TENSORIUM_ENABLE_MPI

/**
 * @brief MPI datatypes and exchange logic for a single Field3D.
 * @tparam T Floating-point type (float or double).
 */
template <typename T> class HaloExchanger {
  public:
    /**
     * @brief Construct halo exchanger for given domain and local grid strides.
     * @param domain The MPI domain decomposition.
     * @param st Strides of the local Field3D arrays.
     * @param local_nx Local physical cells in X (excluding ghosts).
     * @param local_ny Local physical cells in Y.
     * @param local_nz Local physical cells in Z.
     */
    HaloExchanger(const MPIDomain &domain, const tensorium_RG::Strides<T> &st, size_t local_nx,
                  size_t local_ny, size_t local_nz)
        : domain_(domain),
          st_(st),
          nx_(local_nx),
          ny_(local_ny),
          nz_(local_nz),
          ng_(domain.ng()) {

        create_datatypes();
    }

    ~HaloExchanger() { free_datatypes(); }

    HaloExchanger(const HaloExchanger &) = delete;
    HaloExchanger &operator=(const HaloExchanger &) = delete;

    /**
     * @brief Perform blocking halo exchange on a single field.
     * @param field Pointer to the raw field data.
     */
    void exchange_blocking(T *field) {
        // X direction
        exchange_direction(field, 0, X_MINUS, X_PLUS);
        // Y direction
        exchange_direction(field, 1, Y_MINUS, Y_PLUS);
        // Z direction
        exchange_direction(field, 2, Z_MINUS, Z_PLUS);
    }

    /**
     * @brief Start non-blocking halo exchange.
     * @param field Pointer to raw field data.
     * @param requests Output array of MPI requests (12 total: 2 per direction x 3 x 2).
     * @return Number of active requests started.
     */
    int exchange_start(T *field, MPI_Request *requests) {
        int nreq = 0;

        // X direction
        nreq += start_direction(field, 0, X_MINUS, X_PLUS, &requests[nreq]);
        // Y direction
        nreq += start_direction(field, 1, Y_MINUS, Y_PLUS, &requests[nreq]);
        // Z direction
        nreq += start_direction(field, 2, Z_MINUS, Z_PLUS, &requests[nreq]);

        return nreq;
    }

    /**
     * @brief Wait for all non-blocking exchanges to complete.
     * @param nreq Number of requests to wait on.
     * @param requests Array of MPI requests.
     */
    static void exchange_wait(int nreq, MPI_Request *requests) {
        if (nreq > 0) {
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        }
    }

    /**
     * @brief Maximum number of MPI requests for non-blocking exchange.
     */
    static constexpr int MAX_REQUESTS = 12;

  private:
    const MPIDomain         &domain_;
    tensorium_RG::Strides<T> st_;
    size_t                   nx_, ny_, nz_, ng_;

    MPI_Datatype face_types_[3][2][2];
    bool         types_created_ = false;

    MPI_Datatype mpi_type() const {
        if constexpr (std::is_same_v<T, double>) {
            return MPI_DOUBLE;
        } else if constexpr (std::is_same_v<T, float>) {
            return MPI_FLOAT;
        } else {
            static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                          "Only float and double supported");
            return MPI_DATATYPE_NULL;
        }
    }

    void create_datatypes() {
        // Total dimensions including ghosts
        const size_t nx_tot = nx_ + 2 * ng_;
        const size_t ny_tot = ny_ + 2 * ng_;
        const size_t nz_tot = st_.nz_tot; // Already padded for SIMD

        // Strides in elements
        const ptrdiff_t sx = static_cast<ptrdiff_t>(st_.sx);
        const ptrdiff_t sy = static_cast<ptrdiff_t>(st_.sy);
        const ptrdiff_t sz = static_cast<ptrdiff_t>(st_.sz); // Usually 1

        // X-faces: ny_tot x nz_tot x ng slab
        // Send buffer: interior cells at x-boundary
        // Recv buffer: ghost cells
        {
            // YZ plane with thickness ng in X
            // Each "row" in Y has nz_tot elements, rows are sy apart
            // We have ny_tot such rows, and ng such planes (sx apart)

            MPI_Datatype yz_plane;
            MPI_Type_vector(static_cast<int>(ny_tot), // count: ny_tot rows
                            static_cast<int>(nz_tot), // blocklength: nz_tot per row
                            static_cast<int>(sy),     // stride between rows
                            mpi_type(), &yz_plane);

            // Stack ng such planes in X direction
            MPI_Datatype x_slab;
            MPI_Type_create_hvector(static_cast<int>(ng_), // count: ng planes
                                    1,                     // 1 plane each
                                    sx * sizeof(T),        // stride in bytes
                                    yz_plane, &x_slab);
            MPI_Type_commit(&x_slab);
            MPI_Type_free(&yz_plane);

            // All X-face types are the same shape, just different offsets
            for (int sr = 0; sr < 2; ++sr) {
                for (int mp = 0; mp < 2; ++mp) {
                    face_types_[0][sr][mp] = x_slab;
                }
            }
            // Note: We'll handle offsets at send/recv time, not in the type
        }

        // Y-faces: nx_tot x nz_tot x ng slab
        {
            // XZ plane with thickness ng in Y
            MPI_Datatype xz_row;
            MPI_Type_vector(static_cast<int>(nx_tot), // count: nx_tot rows in X
                            static_cast<int>(nz_tot), // blocklength: nz_tot per X-row
                            static_cast<int>(sx),     // stride between X-rows
                            mpi_type(), &xz_row);

            MPI_Datatype y_slab;
            MPI_Type_create_hvector(static_cast<int>(ng_), // ng planes in Y
                                    1, sy * sizeof(T), xz_row, &y_slab);
            MPI_Type_commit(&y_slab);
            MPI_Type_free(&xz_row);

            for (int sr = 0; sr < 2; ++sr) {
                for (int mp = 0; mp < 2; ++mp) {
                    face_types_[1][sr][mp] = y_slab;
                }
            }
        }

        // Z-faces: nx_tot x ny_tot x ng slab
        {
            // XY plane with thickness ng in Z
            // This is contiguous in memory for the Z direction!
            MPI_Datatype xy_plane;
            MPI_Type_vector(static_cast<int>(nx_tot * ny_tot), // all XY points
                            static_cast<int>(ng_),             // ng Z-cells each
                            static_cast<int>(nz_tot),          // stride to next XY point
                            mpi_type(), &xy_plane);
            // Actually, let's do it properly with the correct layout
            MPI_Type_free(&xy_plane);

            // Better approach: row-by-row
            // Each (i,j) point has ng contiguous elements
            MPI_Datatype z_slab;
            MPI_Type_vector(static_cast<int>(nx_tot * ny_tot), static_cast<int>(ng_),
                            static_cast<int>(st_.nz_tot), mpi_type(), &z_slab);
            MPI_Type_commit(&z_slab);

            for (int sr = 0; sr < 2; ++sr) {
                for (int mp = 0; mp < 2; ++mp) {
                    face_types_[2][sr][mp] = z_slab;
                }
            }
        }

        types_created_ = true;
    }

    void free_datatypes() {
        if (!types_created_)
            return;

        // Free unique types (X, Y, Z slabs are each reused)
        MPI_Type_free(&face_types_[0][0][0]);
        MPI_Type_free(&face_types_[1][0][0]);
        MPI_Type_free(&face_types_[2][0][0]);

        types_created_ = false;
    }

    // Compute offset to start of a face region
    size_t face_offset(int dir, bool is_send, bool is_plus) const {
        // Grid layout: [0..ng-1] = lower ghost, [ng..ng+n-1] = interior, [ng+n..] = upper ghost

        if (dir == 0) { // X direction
            if (is_send) {
                // Send from interior cells adjacent to boundary
                if (is_plus) {
                    // Sending to +X neighbor: send from i = ng + nx - ng ... ng + nx - 1
                    return (ng_ + nx_ - ng_) * st_.sx;
                } else {
                    // Sending to -X neighbor: send from i = ng ... ng + ng - 1
                    return ng_ * st_.sx;
                }
            } else {
                // Recv into ghost cells
                if (is_plus) {
                    // Recv from +X neighbor into upper ghost: i = ng + nx ... ng + nx + ng - 1
                    return (ng_ + nx_) * st_.sx;
                } else {
                    // Recv from -X neighbor into lower ghost: i = 0 ... ng - 1
                    return 0;
                }
            }
        } else if (dir == 1) { // Y direction
            if (is_send) {
                if (is_plus) {
                    return (ng_ + ny_ - ng_) * st_.sy;
                } else {
                    return ng_ * st_.sy;
                }
            } else {
                if (is_plus) {
                    return (ng_ + ny_) * st_.sy;
                } else {
                    return 0;
                }
            }
        } else { // Z direction (dir == 2)
            if (is_send) {
                if (is_plus) {
                    return (ng_ + nz_ - ng_) * st_.sz;
                } else {
                    return ng_ * st_.sz;
                }
            } else {
                if (is_plus) {
                    return (ng_ + nz_) * st_.sz;
                } else {
                    return 0;
                }
            }
        }
    }

    void exchange_direction(T *field, int dir, NeighborDir minus, NeighborDir plus) {
        const int rank_minus = domain_.neighbor(minus);
        const int rank_plus = domain_.neighbor(plus);

        const int tag_to_plus = 100 + dir * 2;
        const int tag_to_minus = 100 + dir * 2 + 1;

        MPI_Datatype dtype = face_types_[dir][0][0];

        // Send to minus, recv from plus
        if (rank_minus != MPI_PROC_NULL || rank_plus != MPI_PROC_NULL) {
            T *send_minus = field + face_offset(dir, true, false);
            T *recv_plus = field + face_offset(dir, false, true);

            MPI_Sendrecv(send_minus, 1, dtype, rank_minus, tag_to_minus, recv_plus, 1, dtype,
                         rank_plus, tag_to_minus, domain_.comm(), MPI_STATUS_IGNORE);
        }

        // Send to plus, recv from minus
        if (rank_plus != MPI_PROC_NULL || rank_minus != MPI_PROC_NULL) {
            T *send_plus = field + face_offset(dir, true, true);
            T *recv_minus = field + face_offset(dir, false, false);

            MPI_Sendrecv(send_plus, 1, dtype, rank_plus, tag_to_plus, recv_minus, 1, dtype,
                         rank_minus, tag_to_plus, domain_.comm(), MPI_STATUS_IGNORE);
        }
    }

    int start_direction(T *field, int dir, NeighborDir minus, NeighborDir plus,
                        MPI_Request *requests) {
        const int rank_minus = domain_.neighbor(minus);
        const int rank_plus = domain_.neighbor(plus);

        const int tag_to_plus = 100 + dir * 2;
        const int tag_to_minus = 100 + dir * 2 + 1;

        MPI_Datatype dtype = face_types_[dir][0][0];
        int          nreq = 0;

        // Irecv from plus
        if (rank_plus != MPI_PROC_NULL) {
            T *recv_plus = field + face_offset(dir, false, true);
            MPI_Irecv(recv_plus, 1, dtype, rank_plus, tag_to_minus, domain_.comm(),
                      &requests[nreq++]);
        }

        // Irecv from minus
        if (rank_minus != MPI_PROC_NULL) {
            T *recv_minus = field + face_offset(dir, false, false);
            MPI_Irecv(recv_minus, 1, dtype, rank_minus, tag_to_plus, domain_.comm(),
                      &requests[nreq++]);
        }

        // Isend to minus
        if (rank_minus != MPI_PROC_NULL) {
            T *send_minus = field + face_offset(dir, true, false);
            MPI_Isend(send_minus, 1, dtype, rank_minus, tag_to_minus, domain_.comm(),
                      &requests[nreq++]);
        }

        // Isend to plus
        if (rank_plus != MPI_PROC_NULL) {
            T *send_plus = field + face_offset(dir, true, true);
            MPI_Isend(send_plus, 1, dtype, rank_plus, tag_to_plus, domain_.comm(),
                      &requests[nreq++]);
        }

        return nreq;
    }
};

/**
 * @brief Batch halo exchanger for multiple fields (e.g., all BSSN variables).
 */
template <typename T> class BatchHaloExchanger {
  public:
    struct ExchangeState {
        std::vector<MPI_Request> requests;
        int                      total_requests = 0;
        bool                     active = false;
    };

    BatchHaloExchanger(const MPIDomain &domain, const tensorium_RG::Strides<T> &st, size_t local_nx,
                       size_t local_ny, size_t local_nz)
        : exchanger_(domain, st, local_nx, local_ny, local_nz) {}

    /**
     * @brief Exchange halos for multiple fields sequentially (blocking).
     */
    void exchange_blocking(std::vector<T *> &fields) {
        for (T *field : fields) {
            exchanger_.exchange_blocking(field);
        }
    }

    /**
     * @brief Exchange halos for multiple fields with non-blocking ops.
     */
    void exchange_all(std::vector<T *> &fields) {
        ExchangeState state;
        exchange_start(fields, state);
        exchange_wait(state);
    }

    void exchange_start(std::vector<T *> &fields, ExchangeState &state) {
        const int nfields = static_cast<int>(fields.size());
        state.requests.assign(nfields * HaloExchanger<T>::MAX_REQUESTS, MPI_REQUEST_NULL);
        state.total_requests = 0;

        for (int i = 0; i < nfields; ++i) {
            const int nreq = exchanger_.exchange_start(fields[i], &state.requests[state.total_requests]);
            state.total_requests += nreq;
        }
        state.active = true;
    }

    static void exchange_wait(ExchangeState &state) {
        if (!state.active) {
            return;
        }
        HaloExchanger<T>::exchange_wait(state.total_requests, state.requests.data());
        state.requests.clear();
        state.total_requests = 0;
        state.active = false;
    }

    HaloExchanger<T> &exchanger() { return exchanger_; }

  private:
    HaloExchanger<T> exchanger_;
};

#else // !TENSORIUM_ENABLE_MPI

/**
 * @brief Stub implementation when MPI is disabled.
 */
template <typename T> class HaloExchanger {
  public:
    HaloExchanger(const MPIDomain &, const tensorium_RG::Strides<T> &, size_t, size_t, size_t) {}

    void                 exchange_blocking(T *) {}
    static constexpr int MAX_REQUESTS = 0;
};

template <typename T> class BatchHaloExchanger {
  public:
    struct ExchangeState {};

    BatchHaloExchanger(const MPIDomain &, const tensorium_RG::Strides<T> &, size_t, size_t,
                       size_t) {}

    void exchange_blocking(std::vector<T *> &) {}
    void exchange_all(std::vector<T *> &) {}
    void exchange_start(std::vector<T *> &, ExchangeState &) {}
    static void exchange_wait(ExchangeState &) {}
};

#endif // TENSORIUM_ENABLE_MPI

} // namespace tensorium::mpi
