#pragma once

#include <cstddef>

#ifdef TENSORIUM_BSSN_PROFILE_KERNELS
#    include <array>
#    include <chrono>
#    include <cstdio>
#endif

namespace tensorium_RG::bssn {

enum class KernelTimerId : std::size_t {
    Gamma = 0,
    B,
    Beta,
    Alpha,
    Chi,
    GammaTilde,
    ATilde,
    K,
    Theta,
    Z,
    Count
};

#ifndef TENSORIUM_BSSN_PROFILE_INTERVAL
#    define TENSORIUM_BSSN_PROFILE_INTERVAL 1
#endif

#ifdef TENSORIUM_BSSN_PROFILE_KERNELS

class KernelTimerAggregator {
  public:
    static KernelTimerAggregator &instance() {
        static KernelTimerAggregator inst;
        return inst;
    }

    void add(KernelTimerId id, double seconds) {
        accum_[static_cast<std::size_t>(id)] += seconds;
    }

    void report_and_reset(std::size_t step_index) {
        if (++steps_since_report_ < TENSORIUM_BSSN_PROFILE_INTERVAL)
            return;
        steps_since_report_ = 0;
        static constexpr const char *kNames[] = {"Gamma",      "B",    "Beta",
                                                 "Alpha",      "Chi",  "GammaTilde",
                                                 "A_tilde",    "K",    "Theta",
                                                 "Z"};
        std::printf("[BSSN timers] step=%zu\n", step_index);
        for (std::size_t i = 0; i < accum_.size(); ++i) {
            const double ms = accum_[i] * 1.0e3;
            std::printf("\t%-12s %8.3f ms\n", kNames[i], ms);
            accum_[i] = 0.0;
        }
    }

  private:
    std::array<double, static_cast<std::size_t>(KernelTimerId::Count)> accum_{};
    std::size_t                                                        steps_since_report_ = 0;
};

class KernelTimerScope {
  public:
    explicit KernelTimerScope(KernelTimerId id)
        : id_(id), start_(std::chrono::steady_clock::now()) {}

    ~KernelTimerScope() {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start_;
        KernelTimerAggregator::instance().add(id_, elapsed.count());
    }

  private:
    KernelTimerId                                      id_;
    std::chrono::steady_clock::time_point start_;
};

inline void report_kernel_timers(std::size_t step_index) {
    KernelTimerAggregator::instance().report_and_reset(step_index);
}

#    define BSSN_PROFILE_KERNEL(ID)                                                                    \
        tensorium_RG::bssn::KernelTimerScope timer_scope_##__LINE__(                                    \
            tensorium_RG::bssn::KernelTimerId::ID)

#else

struct KernelTimerScope {
    explicit KernelTimerScope(KernelTimerId) {}
};

inline void report_kernel_timers(std::size_t) {}

#    define BSSN_PROFILE_KERNEL(ID) ((void)0)

#endif

} // namespace tensorium_RG::bssn
