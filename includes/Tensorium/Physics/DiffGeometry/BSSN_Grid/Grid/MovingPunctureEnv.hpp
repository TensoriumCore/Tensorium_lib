#pragma once

#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../InitialData/BSSNInitialData.hpp"
#include "BSSNGridOperations.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>

namespace tensorium_RG::bssn {

struct CircularMomentumSuggestion {
    bool   valid = false;
    double d = 0.0;
    double total_mass = 0.0;
    double reduced_mass = 0.0;
    double p_newtonian = 0.0;
    double p_pn = 0.0;
};

inline CircularMomentumSuggestion suggest_circular_momentum(double m1, double m2, double separation) {
    CircularMomentumSuggestion out{};
    if (!(std::isfinite(m1) && std::isfinite(m2) && std::isfinite(separation)))
        return out;
    if (m1 <= 0.0 || m2 <= 0.0 || separation <= 0.0)
        return out;

    const double d = 2.0 * separation;
    const double M = m1 + m2;
    if (!(std::isfinite(d) && std::isfinite(M)) || d <= 0.0 || M <= 0.0)
        return out;

    const double mu = (m1 * m2) / M;
    const double p_newt = mu * std::sqrt(M / d);
    const double p_pn = 0.295 / std::sqrt(d);
    if (!(std::isfinite(mu) && std::isfinite(p_newt) && std::isfinite(p_pn)))
        return out;

    out.valid = true;
    out.d = d;
    out.total_mass = M;
    out.reduced_mass = mu;
    out.p_newtonian = p_newt;
    out.p_pn = p_pn;
    return out;
}

struct MovingPunctureBoundaryFaces {
    bool rhs_ix1 = true;
    bool rhs_ox1 = true;
    bool rhs_ix2 = true;
    bool rhs_ox2 = true;
    bool rhs_ix3 = false;
    bool rhs_ox3 = true;
    bool rf_ix1 = false;
    bool rf_ox1 = false;
    bool rf_ix2 = false;
    bool rf_ox2 = false;
    bool rf_ix3 = true;
    bool rf_ox3 = false;
};

struct MovingPunctureEnvConfig {
    size_t nx = 96;
    size_t ny = 96;
    size_t nz = 96;
    size_t ng = 6;
    size_t padding = 0;
    size_t steps = 6000;

    double box_length = 62.4;
    double spacing = 62.4 / 96.0;
    double cfl = 0.10;
    double gauge_speed = 1.0;
    int    spatial_derivative_order = 4;

    double mass1 = 0.48847892320123;
    double mass2 = 0.48847892320123;
    double separation = 6.10679;
    double tangential_momentum = 0.0841746;
    double user_tangential_momentum = 0.0841746;
    double radial_momentum = 0.000510846;
    bool   print_suggested_momentum = false;
    bool   auto_circular = false;
    CircularMomentumSuggestion circular_hint{};

    bool   use_interpolated_init = false;
    size_t interp_seed_n = 64;

    GaugeParameters<double> gauge_params{};
    bool                    strict_tp_gauge = false;
    size_t                  projection_stride = 0;
    size_t                  state_log_stride = 10;
    MovingPunctureBoundaryFaces boundary_faces{};
};

namespace detail {

inline std::optional<long> env_long(const char *name) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long v = std::strtol(raw, &end, 10);
        if (end != raw)
            return v;
    }
    return std::nullopt;
}

inline std::optional<double> env_double(const char *name) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const double v = std::strtod(raw, &end);
        if (end != raw && std::isfinite(v))
            return v;
    }
    return std::nullopt;
}

inline bool env_bool_or(const char *name, bool fallback) {
    const auto parsed = env_long(name);
    return parsed ? (*parsed != 0) : fallback;
}

} // namespace detail

inline MovingPunctureEnvConfig load_moving_puncture_env() {
    MovingPunctureEnvConfig cfg;

    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_GRID_N")) {
        if (*parsed > 8) {
            cfg.nx = static_cast<size_t>(*parsed);
            cfg.ny = static_cast<size_t>(*parsed);
            cfg.nz = static_cast<size_t>(*parsed);
        }
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_NX")) {
        if (*parsed > 8)
            cfg.nx = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_NY")) {
        if (*parsed > 8)
            cfg.ny = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_NZ")) {
        if (*parsed > 8)
            cfg.nz = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH")) {
        if (*parsed > 0.0)
            cfg.box_length = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SPACING")) {
        if (*parsed > 0.0)
            cfg.spacing = *parsed;
    } else {
        cfg.spacing = cfg.box_length / static_cast<double>(cfg.nx);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_STEPS")) {
        if (*parsed > 0)
            cfg.steps = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_CFL")) {
        if (*parsed > 0.0)
            cfg.cfl = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_GAUGE_SPEED")) {
        if (*parsed > 0.0)
            cfg.gauge_speed = *parsed;
    }

    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_SPATIAL_ORDER")) {
        if (*parsed == 4 || *parsed == 6)
            cfg.spatial_derivative_order = static_cast<int>(*parsed);
    }

    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_MASS")) {
        if (*parsed > 0.0) {
            cfg.mass1 = *parsed;
            cfg.mass2 = *parsed;
        }
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_MASS1")) {
        if (*parsed > 0.0)
            cfg.mass1 = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_MASS2")) {
        if (*parsed > 0.0)
            cfg.mass2 = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SEPARATION")) {
        if (*parsed > 0.0)
            cfg.separation = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_MOMENTUM")) {
        if (*parsed >= 0.0)
            cfg.tangential_momentum = *parsed;
    }
    cfg.user_tangential_momentum = cfg.tangential_momentum;
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_RADIAL_MOMENTUM")) {
        if (*parsed >= 0.0)
            cfg.radial_momentum = *parsed;
    }

    cfg.print_suggested_momentum =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_PRINT_SUGGESTED_MOMENTUM", false);
    cfg.auto_circular = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_AUTO_CIRCULAR", false);
    cfg.circular_hint = suggest_circular_momentum(cfg.mass1, cfg.mass2, cfg.separation);
    if (cfg.auto_circular && cfg.circular_hint.valid)
        cfg.tangential_momentum = cfg.circular_hint.p_pn;

    cfg.use_interpolated_init =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT", false);
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_INTERP_SEED_N")) {
        if (*parsed >= 24)
            cfg.interp_seed_n = static_cast<size_t>(*parsed);
    }

    GaugeParameters<double> params;
    params.eta = 1.0;
    params.beta_B_coeff = 0.75;
    params.use_direct_shift_rhs = false;
    params.use_shift_advection = false;
    params.shift_Gamma = 0.75;
    params.shift_advect = 0.0;
    params.shift_eta = 1.0;
    params.lapse_oplog = 2.0;
    params.lapse_advect = 1.0;
    params.kappa1 = 0.1;
    params.kappa2 = 0.0;
    params.kappa3 = 1.0;
    params.kappa_z = 1.0;
    params.covariant_z4 = true;
    params.chi_div_floor = 1e-5;
    params.use_theta_in_lapse = true;
    params.ko_sigma = 1.0;
    params.slow_start_lapse = false;
    params.min_lapse_for_K = 1e-4;
    params.max_K_squared = 1e4;
    params.alpha_floor = 1e-4;
    params.chi_floor = 1e-4;
    params.evolve_Z = false;
    params.gamma_damping_uses_metric = false;
    params.apply_rhs_sommerfeld = true;

    cfg.strict_tp_gauge = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_STRICT_TP_GAUGE",
                                              cfg.use_interpolated_init);
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SHIFT_ETA")) {
        if (*parsed >= 0.0) {
            params.eta = *parsed;
            params.shift_eta = *parsed;
        }
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA1")) {
        if (*parsed >= 0.0)
            params.kappa1 = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA2"))
        params.kappa2 = *parsed;
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA3"))
        params.kappa3 = *parsed;
    params.covariant_z4 =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_COVARIANT_Z4", params.covariant_z4);
    params.evolve_Z = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_EVOLVE_Z", params.evolve_Z);
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_CHI_DIV_FLOOR")) {
        if (*parsed > 0.0)
            params.chi_div_floor = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KO_SIGMA")) {
        if (*parsed >= 0.0)
            params.ko_sigma = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_ALPHA_FLOOR")) {
        if (*parsed >= 0.0)
            params.alpha_floor = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_CHI_FLOOR")) {
        if (*parsed >= 0.0)
            params.chi_floor = *parsed;
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_PROJECTION_STRIDE")) {
        if (*parsed > 0)
            cfg.projection_stride = static_cast<size_t>(*parsed);
    }
    if (detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_ULTRA_STABLE", false)) {
        cfg.cfl = std::min(cfg.cfl, 0.07);
        params.eta = std::max(params.eta, 6.0);
        params.shift_eta = std::max(params.shift_eta, 6.0);
        params.kappa1 = std::max(params.kappa1, 0.18);
        params.ko_sigma = std::max(params.ko_sigma, 0.28);
        cfg.projection_stride =
            (cfg.projection_stride == 0) ? size_t(5) : std::min(cfg.projection_stride, size_t(5));
    }

    if (cfg.strict_tp_gauge) {
        params.use_direct_shift_rhs = false;
        params.use_shift_advection = false;
        params.shift_advect = 0.0;
        params.shift_Gamma = 0.75;
        params.beta_B_coeff = 0.75;
        params.eta = 1.0;
        params.shift_eta = 1.0;
        params.lapse_oplog = 2.0;
        params.lapse_advect = 1.0;
        params.use_theta_in_lapse = true;
        params.slow_start_lapse = false;
        params.ko_sigma = 1.0;
        params.alpha_floor = std::max(params.alpha_floor, 1e-4);
        params.chi_floor = std::max(params.chi_floor, 1e-4);
        params.chi_div_floor = std::max(params.chi_div_floor, 1e-4);
        params.min_lapse_for_K = std::max(params.min_lapse_for_K, 1e-4);
    }
    cfg.gauge_params = params;

    auto &bc = cfg.boundary_faces;
    bc.rhs_ix1 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX1", bc.rhs_ix1);
    bc.rhs_ox1 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX1", bc.rhs_ox1);
    bc.rhs_ix2 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX2", bc.rhs_ix2);
    bc.rhs_ox2 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX2", bc.rhs_ox2);
    bc.rhs_ix3 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX3", bc.rhs_ix3);
    bc.rhs_ox3 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX3", bc.rhs_ox3);
    bc.rf_ix1 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX1", bc.rf_ix1);
    bc.rf_ox1 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX1", bc.rf_ox1);
    bc.rf_ix2 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX2", bc.rf_ix2);
    bc.rf_ox2 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX2", bc.rf_ox2);
    bc.rf_ix3 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX3", bc.rf_ix3);
    bc.rf_ox3 = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX3", bc.rf_ox3);

    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_STATE_LOG_STRIDE")) {
        if (*parsed > 0)
            cfg.state_log_stride = static_cast<size_t>(*parsed);
    }

    return cfg;
}

template <typename T> inline void center_cell_centered_origin(BSSNGridSoA<T> &grid) {
    grid.x0 = -0.5 * grid.dx * static_cast<T>(grid.dims.nx) + 0.5 * grid.dx;
    grid.y0 = -0.5 * grid.dy * static_cast<T>(grid.dims.ny) + 0.5 * grid.dy;
    grid.z0 = -0.5 * grid.dz * static_cast<T>(grid.dims.nz) + 0.5 * grid.dz;
}

inline void initialize_moving_puncture_data(BSSNGridSoA<double> &grid,
                                            const MovingPunctureEnvConfig &cfg) {
    const double S1[3] = {0.0, 0.0, 0.0};
    const double S2[3] = {0.0, 0.0, 0.0};

    if (cfg.use_interpolated_init) {
        const double x1 = -cfg.separation;
        const double x2 = cfg.separation;
        const double y = 0.0;
        const double z = 0.0;
        const double P1[3] = {cfg.radial_momentum, -cfg.tangential_momentum, 0.0};
        const double P2[3] = {-cfg.radial_momentum, cfg.tangential_momentum, 0.0};

        tensorium_RG::init::binary_bowen_york_puncture_interpolated_init(
            grid, cfg.mass1, x1, y, z, P1, S1, cfg.mass2, x2, y, z, P2, S2, cfg.interp_seed_n,
            1e-10);
        return;
    }

    const double x = 0.0;
    const double y1 = cfg.separation;
    const double y2 = -cfg.separation;
    const double z = 0.0;
    const double P1[3] = {-cfg.tangential_momentum, -cfg.radial_momentum, 0.0};
    const double P2[3] = {cfg.tangential_momentum, cfg.radial_momentum, 0.0};
    tensorium_RG::init::binary_bowen_york_puncture_init(grid, cfg.mass1, x, y1, z, P1, S1,
                                                        cfg.mass2, x, y2, z, P2, S2, 1e-10);
}

inline const char *moving_puncture_interpolated_mode_name() {
#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
    return "interpolated_twopunctures";
#else
    return "interpolated_seed";
#endif
}

inline void apply_boundary_faces(const MovingPunctureBoundaryFaces &faces) {
    BoundaryRadiative::set_reflective_faces(faces.rf_ix1, faces.rf_ox1, faces.rf_ix2, faces.rf_ox2,
                                            faces.rf_ix3, faces.rf_ox3);
    BoundaryRadiative::set_rhs_sommerfeld_faces(faces.rhs_ix1, faces.rhs_ox1, faces.rhs_ix2,
                                                faces.rhs_ox2, faces.rhs_ix3, faces.rhs_ox3);
}

} // namespace tensorium_RG::bssn
