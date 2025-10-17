#pragma once

#include <cstring>
#include <iostream>
#include <string>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <cpuid.h>
#    define TENSORIUM_X86 1
#elif defined(__aarch64__) || defined(__arm64__)
#    define TENSORIUM_ARM 1
#else
#    define TENSORIUM_FALLBACK 1
#endif

// ─────────────────────────────── CPU Brand ───────────────────────────────
inline std::string get_cpu_brand() {
#if defined(TENSORIUM_X86)
    char         brand[0x40] = {0};
    unsigned int regs[4] = {0};
    for (int i = 0; i < 3; ++i) {
        __cpuid(0x80000002 + i, regs[0], regs[1], regs[2], regs[3]);
        std::memcpy(brand + i * 16, regs, sizeof(regs));
    }
    return std::string(brand);
#elif defined(TENSORIUM_ARM)
// Apple Silicon / ARM64 fallback
// cf. /proc/cpuinfo (Linux) ou sysctl hw.model (macOS)
#    if defined(__APPLE__)
    char   buffer[128];
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) == 0)
        return std::string(buffer);
    if (sysctlbyname("hw.model", &buffer, &size, NULL, 0) == 0)
        return std::string(buffer);
    return "Apple ARM CPU";
#    else
    return "Generic ARM CPU";
#    endif
#else
    return "Unknown CPU";
#endif
}

// ─────────────────────────────── Block size heuristic ───────────────────────────────
inline size_t detect_optimal_block_size() {
    std::string brand = get_cpu_brand();

    if (brand.find("Xeon Phi") != std::string::npos)
        return 256;
    if (brand.find("Xeon") != std::string::npos)
        return 128;
    if (brand.find("Ryzen") != std::string::npos)
        return 96;
    if (brand.find("Apple") != std::string::npos)
        return 64;
    if (brand.find("Core(TM)") != std::string::npos)
        return 128;

#if defined(TENSORIUM_ARM)
    return 64; // safe default for M1/M2
#else
    std::cout << "[detect_optimal_block_size] Unknown CPU brand. Defaulting to 64.\n";
    return 64;
#endif
}
