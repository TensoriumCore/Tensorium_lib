
#pragma once
#include <cstdint>
#include <array>
#include <iostream>

namespace morpheus {

class CacheInfo {
public:
    static int getL1CacheSize();       // in bytes
    static int getL2CacheSize();       // in bytes
    static int getL3CacheSize();       // in bytes
    static int getCacheLineSize();     // in bytes
};

#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>

inline void cpuid(int info[4], int function_id, int subfunction_id = 0) {
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
}


inline int CacheInfo::getCacheLineSize() {
    int info[4];
    cpuid(info, 0x4, 0); // L1 data cache
    int line_size = (info[1] & 0xFFF) + 1;
    return line_size;
}


inline int CacheInfo::getL1CacheSize() {
    int info[4];
    cpuid(info, 0x4, 1);
    int ways = ((info[1] >> 22) & 0x3FF) + 1;
    int partitions = ((info[1] >> 12) & 0x3FF) + 1;
    int line_size = (info[1] & 0xFFF) + 1;
    int sets = info[2] + 1;
    return ways * partitions * line_size * sets;
}

inline int CacheInfo::getL2CacheSize() {
    int info[4];
    cpuid(info, 0x4, 2); 
    int ways = ((info[1] >> 22) & 0x3FF) + 1;
    int partitions = ((info[1] >> 12) & 0x3FF) + 1;
    int line_size = (info[1] & 0xFFF) + 1;
    int sets = info[2] + 1;
    return ways * partitions * line_size * sets;
}

inline int CacheInfo::getL3CacheSize() {
    int info[4];
    cpuid(info, 0x4, 3); // subleaf 3 = L3
    int ways = ((info[1] >> 22) & 0x3FF) + 1;
    int partitions = ((info[1] >> 12) & 0x3FF) + 1;
    int line_size = (info[1] & 0xFFF) + 1;
    int sets = info[2] + 1;
    return ways * partitions * line_size * sets;
}

#else

// For non-x86 platforms (ARM, etc), return dummy values or extend with /sys
inline int CacheInfo::getCacheLineSize() { return 64; }
inline int CacheInfo::getL1CacheSize()   { return 32 * 1024; }
inline int CacheInfo::getL2CacheSize()   { return 256 * 1024; }
inline int CacheInfo::getL3CacheSize()   { return 8 * 1024 * 1024; }

#endif

} // namespace morpheus
