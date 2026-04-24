#pragma once

#include <Tensorium/Backend/Common/Backend.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef TENSORIUM_CUDA
#    include <cuda_runtime.h>
#endif

namespace tensorium::cuda {

enum class MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

inline const char *to_string(MemcpyKind kind) {
    switch (kind) {
    case MemcpyKind::HostToDevice:
        return "host_to_device";
    case MemcpyKind::DeviceToHost:
        return "device_to_host";
    case MemcpyKind::DeviceToDevice:
        return "device_to_device";
    }
    return "unknown";
}

inline const char *compiled_arch_list() {
#ifdef TENSORIUM_CUDA
#    ifdef TENSORIUM_CUDA_ARCH_LIST
    return TENSORIUM_CUDA_ARCH_LIST;
#    endif
    return "unknown";
#else
    return "disabled";
#endif
}

#ifdef TENSORIUM_CUDA

inline cudaMemcpyKind native_memcpy_kind(MemcpyKind kind) {
    switch (kind) {
    case MemcpyKind::HostToDevice:
        return cudaMemcpyHostToDevice;
    case MemcpyKind::DeviceToHost:
        return cudaMemcpyDeviceToHost;
    case MemcpyKind::DeviceToDevice:
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyDefault;
}

inline void throw_if_error(cudaError_t status, const char *context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

inline bool is_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

inline int device_count() {
    int count = 0;
    throw_if_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

inline void set_device(int ordinal) {
    throw_if_error(cudaSetDevice(ordinal), "cudaSetDevice");
}

inline void device_synchronize() {
    throw_if_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

inline backend::DeviceInfo device_info(int ordinal) {
    cudaDeviceProp prop {};
    throw_if_error(cudaGetDeviceProperties(&prop, ordinal), "cudaGetDeviceProperties");

    backend::DeviceInfo info;
    info.backend = backend::Kind::CUDA;
    info.ordinal = ordinal;
    info.name = prop.name;
    info.total_global_memory = prop.totalGlobalMem;
    info.major = prop.major;
    info.minor = prop.minor;
    info.available = true;
    return info;
}

inline void *malloc_bytes(std::size_t bytes) {
    void *ptr = nullptr;
    throw_if_error(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

inline void free_bytes(void *ptr) {
    if (!ptr)
        return;
    throw_if_error(cudaFree(ptr), "cudaFree");
}

inline void memcpy(void *dst, const void *src, std::size_t bytes, MemcpyKind kind) {
    throw_if_error(cudaMemcpy(dst, src, bytes, native_memcpy_kind(kind)), "cudaMemcpy");
}

inline void memset(void *dst, int value, std::size_t bytes) {
    throw_if_error(cudaMemset(dst, value, bytes), "cudaMemset");
}

#else

inline bool is_available() { return false; }

inline int device_count() { return 0; }

inline void set_device(int) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

inline void device_synchronize() {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

inline backend::DeviceInfo device_info(int) {
    return {};
}

inline void *malloc_bytes(std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

inline void free_bytes(void *) {}

inline void memcpy(void *, const void *, std::size_t, MemcpyKind) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

inline void memset(void *, int, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

#endif

} // namespace tensorium::cuda
