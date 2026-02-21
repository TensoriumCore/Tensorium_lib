#pragma once

#include <algorithm>
#include <array>
#include <utility>

#include "Comm.hpp"

namespace tensorium::backend::mpi {

struct Int3 {
    int x = 0;
    int y = 0;
    int z = 0;
};

class CartesianDecomp {
  public:
    CartesianDecomp() = default;

    CartesianDecomp(const Comm &world, Int3 global_cells, Int3 dims_hint = {0, 0, 0},
                    std::array<int, 3> periods = {0, 0, 0})
        : world_(world), comm_(world), global_cells_(global_cells), periods_(periods) {
        initialize(dims_hint);
    }

    ~CartesianDecomp() { release(); }

    CartesianDecomp(const CartesianDecomp &) = delete;
    CartesianDecomp &operator=(const CartesianDecomp &) = delete;

    CartesianDecomp(CartesianDecomp &&other) noexcept { move_from(std::move(other)); }

    CartesianDecomp &operator=(CartesianDecomp &&other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    const Comm &world() const noexcept { return world_; }
    const Comm &comm() const noexcept { return comm_; }

    Int3 global_cells() const noexcept { return global_cells_; }
    Int3 local_offset() const noexcept { return local_begin_; }
    Int3 local_cells() const noexcept { return local_cells_; }
    Int3 proc_dims() const noexcept { return proc_dims_; }
    Int3 coords() const noexcept { return coords_; }

    int neighbor_minus(int axis) const noexcept { return neighbors_minus_[std::clamp(axis, 0, 2)]; }
    int neighbor_plus(int axis) const noexcept { return neighbors_plus_[std::clamp(axis, 0, 2)]; }

  private:
    static void split_1d(int n_global, int n_proc, int coord, int &begin, int &end) {
        const int base = n_global / n_proc;
        const int rem = n_global % n_proc;
        if (coord < rem) {
            begin = coord * (base + 1);
            end = begin + (base + 1);
        } else {
            begin = rem * (base + 1) + (coord - rem) * base;
            end = begin + base;
        }
    }

    void initialize(Int3 dims_hint) {
        proc_dims_ = {1, 1, 1};
        coords_ = {0, 0, 0};
        neighbors_minus_ = {{-1, -1, -1}};
        neighbors_plus_ = {{-1, -1, -1}};

#if TENSORIUM_BACKEND_MPI_ENABLED
        int dims[3] = {std::max(dims_hint.x, 0), std::max(dims_hint.y, 0),
                       std::max(dims_hint.z, 0)};
        MPI_Dims_create(world_.size(), 3, dims);

        int periods[3] = {periods_[0], periods_[1], periods_[2]};
        MPI_Comm cart = MPI_COMM_NULL;
        MPI_Cart_create(world_.native(), 3, dims, periods, 0, &cart);
        if (cart != MPI_COMM_NULL) {
            cart_comm_ = cart;
            owns_cart_comm_ = true;
            comm_ = Comm(cart_comm_);
        }

        proc_dims_ = {dims[0], dims[1], dims[2]};

        int c[3] = {0, 0, 0};
        MPI_Cart_coords(comm_.native(), comm_.rank(), 3, c);
        coords_ = {c[0], c[1], c[2]};

        for (int axis = 0; axis < 3; ++axis) {
            int src = -1;
            int dst = -1;
            MPI_Cart_shift(comm_.native(), axis, 1, &src, &dst);
            neighbors_minus_[axis] = src;
            neighbors_plus_[axis] = dst;
        }
#endif

        int bx = 0, ex = 0;
        int by = 0, ey = 0;
        int bz = 0, ez = 0;
        split_1d(global_cells_.x, proc_dims_.x, coords_.x, bx, ex);
        split_1d(global_cells_.y, proc_dims_.y, coords_.y, by, ey);
        split_1d(global_cells_.z, proc_dims_.z, coords_.z, bz, ez);
        local_begin_ = {bx, by, bz};
        local_cells_ = {ex - bx, ey - by, ez - bz};
    }

    void release() {
#if TENSORIUM_BACKEND_MPI_ENABLED
        if (owns_cart_comm_ && cart_comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_comm_);
            cart_comm_ = MPI_COMM_NULL;
            owns_cart_comm_ = false;
        }
#endif
    }

    void move_from(CartesianDecomp &&other) {
        world_ = other.world_;
        comm_ = other.comm_;
        global_cells_ = other.global_cells_;
        periods_ = other.periods_;
        proc_dims_ = other.proc_dims_;
        coords_ = other.coords_;
        local_begin_ = other.local_begin_;
        local_cells_ = other.local_cells_;
        neighbors_minus_ = other.neighbors_minus_;
        neighbors_plus_ = other.neighbors_plus_;
#if TENSORIUM_BACKEND_MPI_ENABLED
        cart_comm_ = other.cart_comm_;
        owns_cart_comm_ = other.owns_cart_comm_;
        other.cart_comm_ = MPI_COMM_NULL;
        other.owns_cart_comm_ = false;
#endif
    }

    Comm world_{};
    Comm comm_{};

    Int3 global_cells_{};
    std::array<int, 3> periods_ = {{0, 0, 0}};

    Int3 proc_dims_{1, 1, 1};
    Int3 coords_{0, 0, 0};
    Int3 local_begin_{0, 0, 0};
    Int3 local_cells_{0, 0, 0};
    std::array<int, 3> neighbors_minus_ = {{-1, -1, -1}};
    std::array<int, 3> neighbors_plus_ = {{-1, -1, -1}};

#if TENSORIUM_BACKEND_MPI_ENABLED
    MPI_Comm cart_comm_ = MPI_COMM_NULL;
    bool owns_cart_comm_ = false;
#endif
};

} // namespace tensorium::backend::mpi

