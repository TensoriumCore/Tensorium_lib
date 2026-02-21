#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "CartesianDecomp.hpp"

namespace tensorium::backend::mpi {

class HaloExchange3D {
  public:
    HaloExchange3D(const CartesianDecomp &decomp, int nx, int ny, int nz, int ng)
        : decomp_(decomp), nx_(nx), ny_(ny), nz_(nz), ng_(ng) {}

    template <typename T>
    void exchange(T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz) const {
        if (decomp_.comm().size() <= 1)
            return;
        exchange_x(data, sx, sy, sz);
        exchange_y(data, sx, sy, sz);
        exchange_z(data, sx, sy, sz);
    }

  private:
    template <typename T>
    static void pack_block(const T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz,
                           int i0, int ni, int j0, int nj, int k0, int nk, std::vector<T> &out) {
        out.resize(static_cast<size_t>(ni) * static_cast<size_t>(nj) * static_cast<size_t>(nk));
        size_t p = 0;
        for (int i = i0; i < i0 + ni; ++i)
            for (int j = j0; j < j0 + nj; ++j)
                for (int k = k0; k < k0 + nk; ++k)
                    out[p++] = data[static_cast<std::ptrdiff_t>(i) * sx +
                                    static_cast<std::ptrdiff_t>(j) * sy +
                                    static_cast<std::ptrdiff_t>(k) * sz];
    }

    template <typename T>
    static void unpack_block(T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz, int i0,
                             int ni, int j0, int nj, int k0, int nk, const std::vector<T> &in) {
        size_t p = 0;
        for (int i = i0; i < i0 + ni; ++i)
            for (int j = j0; j < j0 + nj; ++j)
                for (int k = k0; k < k0 + nk; ++k)
                    data[static_cast<std::ptrdiff_t>(i) * sx + static_cast<std::ptrdiff_t>(j) * sy +
                         static_cast<std::ptrdiff_t>(k) * sz] = in[p++];
    }

    template <typename T>
    void exchange_x(T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz) const {
        const int nbr_m = decomp_.neighbor_minus(0);
        const int nbr_p = decomp_.neighbor_plus(0);
        if (nbr_m < 0 && nbr_p < 0)
            return;

        std::vector<T> send_m, send_p, recv_m, recv_p;
        std::vector<Comm::Request> reqs;
        reqs.reserve(4);

        if (nbr_m >= 0) {
            pack_block(data, sx, sy, sz, ng_, ng_, ng_, ny_, ng_, nz_, send_m);
            recv_m.resize(send_m.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_m.data(), static_cast<int>(recv_m.size() * sizeof(T)),
                                       nbr_m, kTagXMinusToPlus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            pack_block(data, sx, sy, sz, ng_ + nx_ - ng_, ng_, ng_, ny_, ng_, nz_, send_p);
            recv_p.resize(send_p.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_p.data(), static_cast<int>(recv_p.size() * sizeof(T)),
                                       nbr_p, kTagXPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_m >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_m.data(), static_cast<int>(send_m.size() * sizeof(T)),
                                       nbr_m, kTagXPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_p.data(), static_cast<int>(send_p.size() * sizeof(T)),
                                       nbr_p, kTagXMinusToPlus, r);
            reqs.push_back(r);
        }

        decomp_.comm().wait_all(reqs);

        if (nbr_m >= 0)
            unpack_block(data, sx, sy, sz, 0, ng_, ng_, ny_, ng_, nz_, recv_m);
        if (nbr_p >= 0)
            unpack_block(data, sx, sy, sz, ng_ + nx_, ng_, ng_, ny_, ng_, nz_, recv_p);
    }

    template <typename T>
    void exchange_y(T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz) const {
        const int nbr_m = decomp_.neighbor_minus(1);
        const int nbr_p = decomp_.neighbor_plus(1);
        if (nbr_m < 0 && nbr_p < 0)
            return;

        std::vector<T> send_m, send_p, recv_m, recv_p;
        std::vector<Comm::Request> reqs;
        reqs.reserve(4);

        if (nbr_m >= 0) {
            pack_block(data, sx, sy, sz, ng_, nx_, ng_, ng_, ng_, nz_, send_m);
            recv_m.resize(send_m.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_m.data(), static_cast<int>(recv_m.size() * sizeof(T)),
                                       nbr_m, kTagYMinusToPlus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            pack_block(data, sx, sy, sz, ng_, nx_, ng_ + ny_ - ng_, ng_, ng_, nz_, send_p);
            recv_p.resize(send_p.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_p.data(), static_cast<int>(recv_p.size() * sizeof(T)),
                                       nbr_p, kTagYPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_m >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_m.data(), static_cast<int>(send_m.size() * sizeof(T)),
                                       nbr_m, kTagYPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_p.data(), static_cast<int>(send_p.size() * sizeof(T)),
                                       nbr_p, kTagYMinusToPlus, r);
            reqs.push_back(r);
        }

        decomp_.comm().wait_all(reqs);

        if (nbr_m >= 0)
            unpack_block(data, sx, sy, sz, ng_, nx_, 0, ng_, ng_, nz_, recv_m);
        if (nbr_p >= 0)
            unpack_block(data, sx, sy, sz, ng_, nx_, ng_ + ny_, ng_, ng_, nz_, recv_p);
    }

    template <typename T>
    void exchange_z(T *data, std::ptrdiff_t sx, std::ptrdiff_t sy, std::ptrdiff_t sz) const {
        const int nbr_m = decomp_.neighbor_minus(2);
        const int nbr_p = decomp_.neighbor_plus(2);
        if (nbr_m < 0 && nbr_p < 0)
            return;

        std::vector<T> send_m, send_p, recv_m, recv_p;
        std::vector<Comm::Request> reqs;
        reqs.reserve(4);

        if (nbr_m >= 0) {
            pack_block(data, sx, sy, sz, ng_, nx_, ng_, ny_, ng_, ng_, send_m);
            recv_m.resize(send_m.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_m.data(), static_cast<int>(recv_m.size() * sizeof(T)),
                                       nbr_m, kTagZMinusToPlus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            pack_block(data, sx, sy, sz, ng_, nx_, ng_, ny_, ng_ + nz_ - ng_, ng_, send_p);
            recv_p.resize(send_p.size());
            Comm::Request r{};
            decomp_.comm().irecv_bytes(recv_p.data(), static_cast<int>(recv_p.size() * sizeof(T)),
                                       nbr_p, kTagZPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_m >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_m.data(), static_cast<int>(send_m.size() * sizeof(T)),
                                       nbr_m, kTagZPlusToMinus, r);
            reqs.push_back(r);
        }
        if (nbr_p >= 0) {
            Comm::Request r{};
            decomp_.comm().isend_bytes(send_p.data(), static_cast<int>(send_p.size() * sizeof(T)),
                                       nbr_p, kTagZMinusToPlus, r);
            reqs.push_back(r);
        }

        decomp_.comm().wait_all(reqs);

        if (nbr_m >= 0)
            unpack_block(data, sx, sy, sz, ng_, nx_, ng_, ny_, 0, ng_, recv_m);
        if (nbr_p >= 0)
            unpack_block(data, sx, sy, sz, ng_, nx_, ng_, ny_, ng_ + nz_, ng_, recv_p);
    }

    static constexpr int kTagXMinusToPlus = 100;
    static constexpr int kTagXPlusToMinus = 101;
    static constexpr int kTagYMinusToPlus = 110;
    static constexpr int kTagYPlusToMinus = 111;
    static constexpr int kTagZMinusToPlus = 120;
    static constexpr int kTagZPlusToMinus = 121;

    const CartesianDecomp &decomp_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int ng_ = 0;
};

} // namespace tensorium::backend::mpi

