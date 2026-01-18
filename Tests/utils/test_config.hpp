#pragma once

#include <cstddef>
#include <string>
#include <cstddef>
#include <string>

namespace tensorium::tests {

enum class BoundaryType { Clamp, Sponge };

struct StabilityRunConfig {
    size_t nx = 32;
    size_t ny = 32;
    size_t nz = 32;
    size_t ng = 6;
    size_t padding = 4;
    size_t steps = 150;
    double spacing = 0.5;
    double cfl = 0.15;
    double gauge_factor = 1.0;
    double gauge_eta = 2.0;
    double gauge_beta_coeff = 0.5;
    bool freeze_gauge = true;
    bool simulate_only = true;
    BoundaryType boundary = BoundaryType::Clamp;
    std::string log_path;
};

struct RunSummary {
    double max_H = 0.0;
    double max_gamma = 0.0;
    double min_alpha = 1.0;
    double max_det_drift = 0.0;
    double max_trace_A = 0.0;
};

} // namespace tensorium::tests
