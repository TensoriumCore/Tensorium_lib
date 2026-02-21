#pragma once

#include <type_traits>
#include <vector>

#if defined(TENSORIUM_USE_MPI) || defined(MORPHEUS_USE_MPI)
#    include <mpi.h>
#    define TENSORIUM_BACKEND_MPI_ENABLED 1
#else
#    define TENSORIUM_BACKEND_MPI_ENABLED 0
#endif

namespace tensorium::backend::mpi {

namespace detail {

#if TENSORIUM_BACKEND_MPI_ENABLED
template <typename T> struct MPITypeMap {
    static constexpr bool supported = false;
};

template <> struct MPITypeMap<float> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_FLOAT; }
};
template <> struct MPITypeMap<double> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_DOUBLE; }
};
template <> struct MPITypeMap<int> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_INT; }
};
template <> struct MPITypeMap<unsigned int> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_UNSIGNED; }
};
template <> struct MPITypeMap<long> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_LONG; }
};
template <> struct MPITypeMap<unsigned long> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_UNSIGNED_LONG; }
};
template <> struct MPITypeMap<long long> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_LONG_LONG; }
};
template <> struct MPITypeMap<unsigned long long> {
    static constexpr bool supported = true;
    static MPI_Datatype datatype() { return MPI_UNSIGNED_LONG_LONG; }
};
#endif

} // namespace detail

class Comm {
  public:
#if TENSORIUM_BACKEND_MPI_ENABLED
    using NativeComm = MPI_Comm;
    using Request = MPI_Request;
    static constexpr bool enabled = true;

    explicit Comm(NativeComm comm = MPI_COMM_WORLD) : comm_(comm) {}

    NativeComm native() const noexcept { return comm_; }
#else
    using NativeComm = int;
    struct Request {
        int dummy = 0;
    };
    static constexpr bool enabled = false;

    explicit Comm(NativeComm = 0) {}

    NativeComm native() const noexcept { return 0; }
#endif

    int rank() const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        int r = 0;
        MPI_Comm_rank(comm_, &r);
        return r;
#else
        return 0;
#endif
    }

    int size() const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        int s = 1;
        MPI_Comm_size(comm_, &s);
        return s;
#else
        return 1;
#endif
    }

    void barrier() const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        MPI_Barrier(comm_);
#endif
    }

    template <typename T> T allreduce_sum(T value) const noexcept {
        static_assert(std::is_arithmetic<T>::value, "allreduce_sum requires arithmetic T");
#if TENSORIUM_BACKEND_MPI_ENABLED
        static_assert(detail::MPITypeMap<T>::supported, "Unsupported MPI allreduce type");
        T out{};
        MPI_Allreduce(&value, &out, 1, detail::MPITypeMap<T>::datatype(), MPI_SUM, comm_);
        return out;
#else
        return value;
#endif
    }

    template <typename T> T allreduce_min(T value) const noexcept {
        static_assert(std::is_arithmetic<T>::value, "allreduce_min requires arithmetic T");
#if TENSORIUM_BACKEND_MPI_ENABLED
        static_assert(detail::MPITypeMap<T>::supported, "Unsupported MPI allreduce type");
        T out{};
        MPI_Allreduce(&value, &out, 1, detail::MPITypeMap<T>::datatype(), MPI_MIN, comm_);
        return out;
#else
        return value;
#endif
    }

    template <typename T> T allreduce_max(T value) const noexcept {
        static_assert(std::is_arithmetic<T>::value, "allreduce_max requires arithmetic T");
#if TENSORIUM_BACKEND_MPI_ENABLED
        static_assert(detail::MPITypeMap<T>::supported, "Unsupported MPI allreduce type");
        T out{};
        MPI_Allreduce(&value, &out, 1, detail::MPITypeMap<T>::datatype(), MPI_MAX, comm_);
        return out;
#else
        return value;
#endif
    }

    void isend_bytes(const void *data, int bytes, int dest, int tag, Request &request) const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        if (dest < 0) {
            request = MPI_REQUEST_NULL;
            return;
        }
        MPI_Isend(data, bytes, MPI_BYTE, dest, tag, comm_, &request);
#else
        (void)data;
        (void)bytes;
        (void)dest;
        (void)tag;
        (void)request;
#endif
    }

    void irecv_bytes(void *data, int bytes, int src, int tag, Request &request) const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        if (src < 0) {
            request = MPI_REQUEST_NULL;
            return;
        }
        MPI_Irecv(data, bytes, MPI_BYTE, src, tag, comm_, &request);
#else
        (void)data;
        (void)bytes;
        (void)src;
        (void)tag;
        (void)request;
#endif
    }

    void wait_all(std::vector<Request> &requests) const noexcept {
#if TENSORIUM_BACKEND_MPI_ENABLED
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
#else
        (void)requests;
#endif
        requests.clear();
    }

  private:
#if TENSORIUM_BACKEND_MPI_ENABLED
    NativeComm comm_;
#endif
};

} // namespace tensorium::backend::mpi

