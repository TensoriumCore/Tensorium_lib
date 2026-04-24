#pragma once

#include <cstddef>
#include <string>

namespace tensorium::backend {

enum class Kind {
    CPU,
    CUDA
};

enum class MemorySpace {
    Host,
    Device,
    Unified
};

struct Options {
    Kind backend = Kind::CPU;
    int  device_ordinal = 0;
    bool allow_async = false;
};

struct DeviceInfo {
    Kind        backend = Kind::CPU;
    int         ordinal = -1;
    std::string name;
    std::size_t total_global_memory = 0;
    int         major = 0;
    int         minor = 0;
    bool        available = false;
};

inline const char *to_string(Kind kind) {
    switch (kind) {
    case Kind::CPU:
        return "cpu";
    case Kind::CUDA:
        return "cuda";
    }
    return "unknown";
}

inline const char *to_string(MemorySpace space) {
    switch (space) {
    case MemorySpace::Host:
        return "host";
    case MemorySpace::Device:
        return "device";
    case MemorySpace::Unified:
        return "unified";
    }
    return "unknown";
}

} // namespace tensorium::backend
