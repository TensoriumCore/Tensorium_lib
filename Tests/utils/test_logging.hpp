#pragma once

#include "test_config.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace tensorium::tests {

class TestLogger {
  public:
    explicit TestLogger(const std::string &path) {
        if (path.empty())
            return;
        std::filesystem::path out_path(path);
        if (out_path.has_parent_path())
            std::filesystem::create_directories(out_path.parent_path());
        stream_.open(out_path, std::ios::out | std::ios::trunc);
    }

    bool enabled() const noexcept { return stream_.is_open(); }

    void write_header() {
        if (!enabled())
            return;
        stream_ << "step,dt,max_H,max_gamma,min_alpha,max_det_drift,max_trace_A\n";
    }

    void write_step(size_t step, double dt, const tensorium_RG::bssn::ConstraintMonitorStats &stats,
                    double max_gamma, double min_alpha) {
        if (!enabled())
            return;
        stream_ << step << ',' << dt << ',' << stats.max_H << ',' << max_gamma << ',' << min_alpha
                << ',' << stats.max_det_drift << ',' << stats.max_trace_A << '\n';
    }

  private:
    std::ofstream stream_;
};

} // namespace tensorium::tests
