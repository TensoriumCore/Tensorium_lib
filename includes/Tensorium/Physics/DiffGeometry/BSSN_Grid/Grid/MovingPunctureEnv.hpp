#pragma once

#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../InitialData/BSSNInitialData.hpp"
#include "BSSNGridOperations.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

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
    bool rhs_ix3 = true;
    bool rhs_ox3 = true;
    bool rf_ix1 = false;
    bool rf_ox1 = false;
    bool rf_ix2 = false;
    bool rf_ox2 = false;
    bool rf_ix3 = false;
    bool rf_ox3 = false;
};

struct MovingPunctureSpongeConfig {
    bool enabled = true;
    size_t width = 12;
    double strength = 4.0;
    double exponent = 2.0;
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
    MovingPunctureSpongeConfig sponge{};
    size_t                  radiative_collar_width = 4;
    size_t                  ko_boundary_width = 4;
    double                  ko_boundary_floor = 0.0;
    bool                    allow_reflective_bc = false;
    bool                    fail_on_gauge_bc_mismatch = false;
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
        auto parse_exact = [](const char *text) -> std::optional<double> {
            char *end = nullptr;
            const double v = std::strtod(text, &end);
            if (end == text || !std::isfinite(v))
                return std::nullopt;
            while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end)))
                ++end;
            if (*end == '\0')
                return v;
            return std::nullopt;
        };

        if (const auto parsed = parse_exact(raw))
            return parsed;

        std::string normalized(raw);
        if (normalized.find(',') != std::string::npos) {
            std::replace(normalized.begin(), normalized.end(), ',', '.');
            if (const auto parsed = parse_exact(normalized.c_str()))
                return parsed;
        }
    }
    return std::nullopt;
}

inline bool env_bool_or(const char *name, bool fallback) {
    const auto parsed = env_long(name);
    return parsed ? (*parsed != 0) : fallback;
}

} // namespace detail

inline const char *boundary_face_name(int axis, bool outer) {
    switch (std::clamp(axis, 0, 2)) {
    case 0:
        return outer ? "ox1" : "ix1";
    case 1:
        return outer ? "ox2" : "ix2";
    default:
        return outer ? "ox3" : "ix3";
    }
}

inline bool boundary_face_rhs_enabled(const MovingPunctureBoundaryFaces &faces, int axis, bool outer) {
    switch (std::clamp(axis, 0, 2)) {
    case 0:
        return outer ? faces.rhs_ox1 : faces.rhs_ix1;
    case 1:
        return outer ? faces.rhs_ox2 : faces.rhs_ix2;
    default:
        return outer ? faces.rhs_ox3 : faces.rhs_ix3;
    }
}

inline bool boundary_face_reflective_enabled(const MovingPunctureBoundaryFaces &faces, int axis,
                                             bool outer) {
    switch (std::clamp(axis, 0, 2)) {
    case 0:
        return outer ? faces.rf_ox1 : faces.rf_ix1;
    case 1:
        return outer ? faces.rf_ox2 : faces.rf_ix2;
    default:
        return outer ? faces.rf_ox3 : faces.rf_ix3;
    }
}

inline std::string describe_boundary_face_mode(const MovingPunctureBoundaryFaces &faces,
                                               const MovingPunctureSpongeConfig &sponge, int axis,
                                               bool outer) {
    const bool reflective = boundary_face_reflective_enabled(faces, axis, outer);
    const bool sommerfeld = boundary_face_rhs_enabled(faces, axis, outer);

    std::ostringstream oss;
    oss << boundary_face_name(axis, outer) << ":";
    if (reflective) {
        oss << "reflective";
        return oss.str();
    }
    if (sommerfeld)
        oss << "radiative";
    else
        oss << "outflow";
    if (sponge.enabled && sponge.width > 0 && sponge.strength > 0.0)
        oss << "+sponge";
    return oss.str();
}

inline void validate_moving_puncture_boundary_faces(MovingPunctureBoundaryFaces &faces,
                                                    bool allow_reflective_bc) {
    auto normalize_face = [&](bool &rhs, bool &reflective, const char *name) {
        if (reflective && !allow_reflective_bc) {
            std::ostringstream oss;
            oss << "Reflective moving-puncture boundary requested on " << name
                << ". This is disabled by default because it reflects gauge/constraint content "
                   "back into the domain. Set TENSORIUM_MOVING_PUNCTURE_ALLOW_REFLECTIVE_BC=1 "
                   "to opt in explicitly.";
            throw std::runtime_error(oss.str());
        }
        if (reflective && rhs) {
            std::fprintf(stderr,
                         "[bc][warn] %s requested with both reflective and RHS Sommerfeld; "
                         "disabling RHS Sommerfeld because reflective dominates.\n",
                         name);
            rhs = false;
        }
        if (!reflective && !rhs) {
            std::fprintf(stderr,
                         "[bc][warn] %s uses fallback outflow ghosts without RHS Sommerfeld.\n",
                         name);
        }
        if (reflective) {
            std::fprintf(stderr,
                         "[bc][warn] %s is reflective. This is intended only for dedicated "
                         "boundary tests, not production moving-puncture runs.\n",
                         name);
        }
    };

    normalize_face(faces.rhs_ix1, faces.rf_ix1, "ix1");
    normalize_face(faces.rhs_ox1, faces.rf_ox1, "ox1");
    normalize_face(faces.rhs_ix2, faces.rf_ix2, "ix2");
    normalize_face(faces.rhs_ox2, faces.rf_ox2, "ox2");
    normalize_face(faces.rhs_ix3, faces.rf_ix3, "ix3");
    normalize_face(faces.rhs_ox3, faces.rf_ox3, "ox3");
}

inline MovingPunctureEnvConfig load_moving_puncture_env() {
    MovingPunctureEnvConfig cfg;
    cfg.sponge.width = std::max<size_t>(cfg.ng, size_t(8));
    cfg.ko_boundary_width = size_t(4);

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
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SHIFT_GAMMA")) {
        params.shift_Gamma = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA1")) {
        if (*parsed >= 0.0)
            params.kappa1 = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA2"))
        params.kappa2 = *parsed;
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA3"))
        params.kappa3 = *parsed;
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KAPPA_Z"))
        params.kappa_z = *parsed;
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

    params.use_direct_shift_rhs = detail::env_bool_or(
        "TENSORIUM_MOVING_PUNCTURE_USE_DIRECT_SHIFT_RHS", params.use_direct_shift_rhs);
    params.use_shift_advection = detail::env_bool_or(
        "TENSORIUM_MOVING_PUNCTURE_USE_SHIFT_ADVECTION", params.use_shift_advection);
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SHIFT_ADVECT")) {
        params.shift_advect = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_LAPSE_ADVECT")) {
        params.lapse_advect = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_BETA_B_COEFF")) {
        params.beta_B_coeff = *parsed;
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

    cfg.allow_reflective_bc =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_ALLOW_REFLECTIVE_BC", false);
    validate_moving_puncture_boundary_faces(cfg.boundary_faces, cfg.allow_reflective_bc);
    cfg.fail_on_gauge_bc_mismatch =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_FAIL_ON_GAUGE_BC_MISMATCH", false);

    cfg.sponge.enabled = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_ENABLE_SPONGE",
                                             cfg.sponge.enabled);
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_SPONGE_WIDTH")) {
        if (*parsed >= 0)
            cfg.sponge.width = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SPONGE_STRENGTH")) {
        if (*parsed >= 0.0)
            cfg.sponge.strength = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_SPONGE_EXPONENT")) {
        if (*parsed >= 1.0)
            cfg.sponge.exponent = *parsed;
    }

    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_RADIATIVE_COLLAR_WIDTH")) {
        if (*parsed > 0)
            cfg.radiative_collar_width = static_cast<size_t>(*parsed);
    }

    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_KO_BOUNDARY_WIDTH")) {
        if (*parsed >= 0)
            cfg.ko_boundary_width = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KO_BOUNDARY_FLOOR")) {
        cfg.ko_boundary_floor = std::clamp(*parsed, 0.0, 1.0);
    }

    if (params.evolve_Z) {
        std::fprintf(stderr,
                     "[z4c][warn] TENSORIUM_MOVING_PUNCTURE_EVOLVE_Z=1 keeps Z_i synchronized "
                     "from Gamma each RHS assembly; no independent Z_i characteristic boundary "
                     "system is implemented.\n");
    }

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

        tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init(
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
    return "interpolated_twopunctures";
}

inline void apply_boundary_faces(const MovingPunctureBoundaryFaces &faces) {
    BoundaryRadiative::set_reflective_faces(faces.rf_ix1, faces.rf_ox1, faces.rf_ix2, faces.rf_ox2,
                                            faces.rf_ix3, faces.rf_ox3);
    BoundaryRadiative::set_rhs_sommerfeld_faces(faces.rhs_ix1, faces.rhs_ox1, faces.rhs_ix2,
                                                faces.rhs_ox2, faces.rhs_ix3, faces.rhs_ox3);
}

inline void apply_boundary_configuration(const MovingPunctureEnvConfig &cfg) {
    apply_boundary_faces(cfg.boundary_faces);
    BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0, std::sqrt(2.0));
    BoundaryRadiative::set_rhs_collar_width(cfg.radiative_collar_width);
    BoundaryRadiative::set_sponge(cfg.sponge.enabled, cfg.sponge.width, cfg.sponge.strength,
                                  cfg.sponge.exponent);
    configure_ko_boundary_taper(cfg.ko_boundary_width, cfg.ko_boundary_floor);
}

struct MovingPunctureBoundaryCharacteristicSummary {
    double alpha_min = std::numeric_limits<double>::infinity();
    double alpha_max = 0.0;
    double beta_normal_min = std::numeric_limits<double>::infinity();
    double beta_normal_max = -std::numeric_limits<double>::infinity();
    double gauge_speed_max = 0.0;
    double configured_gauge_speed = 0.0;
    size_t samples = 0;
    bool   active = false;
    bool   radiative = false;
    bool   reflective = false;
    bool   mismatch = false;
};

template <typename T>
inline MovingPunctureBoundaryCharacteristicSummary
summarize_boundary_face_characteristics(const BSSNGridSoA<T> &grid,
                                        const MovingPunctureEnvConfig &cfg, int axis,
                                        bool outer) {
    MovingPunctureBoundaryCharacteristicSummary out{};
    out.active = BoundaryRadiative::active_enabled(axis, outer);
    out.radiative = boundary_face_rhs_enabled(cfg.boundary_faces, axis, outer);
    out.reflective = boundary_face_reflective_enabled(cfg.boundary_faces, axis, outer);
    if (!out.active)
        return out;

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    if (I0 >= I1 || J0 >= J1 || K0 >= K1)
        return out;

    const size_t extent = (axis == 0) ? (I1 - I0) : (axis == 1 ? (J1 - J0) : (K1 - K0));
    const size_t layers = std::max<size_t>(
        size_t(1), std::min(extent, std::max<size_t>(cfg.radiative_collar_width, size_t(1))));
    const double alpha_floor = std::max<double>(static_cast<double>(cfg.gauge_params.alpha_floor),
                                                1.0e-6);
    const double normal_sign = outer ? 1.0 : -1.0;

    auto accumulate = [&](size_t i, size_t j, size_t k) {
        const size_t idx = grid.alpha.idx(i, j, k);
        const double alpha = std::max<double>(static_cast<double>(grid.alpha.ptr()[idx]), alpha_floor);
        const double beta_n = normal_sign * static_cast<double>(grid.beta[axis].ptr()[idx]);
        const double gauge_speed = std::sqrt(2.0 / alpha);

        out.alpha_min = std::min(out.alpha_min, alpha);
        out.alpha_max = std::max(out.alpha_max, alpha);
        out.beta_normal_min = std::min(out.beta_normal_min, beta_n);
        out.beta_normal_max = std::max(out.beta_normal_max, beta_n);
        out.gauge_speed_max = std::max(out.gauge_speed_max, gauge_speed);
        ++out.samples;
    };

    for (size_t layer = 0; layer < layers; ++layer) {
        if (axis == 0) {
            const size_t i = outer ? (I1 - 1 - layer) : (I0 + layer);
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k)
                    accumulate(i, j, k);
        } else if (axis == 1) {
            const size_t j = outer ? (J1 - 1 - layer) : (J0 + layer);
            for (size_t i = I0; i < I1; ++i)
                for (size_t k = K0; k < K1; ++k)
                    accumulate(i, j, k);
        } else {
            const size_t k = outer ? (K1 - 1 - layer) : (K0 + layer);
            for (size_t i = I0; i < I1; ++i)
                for (size_t j = J0; j < J1; ++j)
                    accumulate(i, j, k);
        }
    }

    if (out.samples == 0) {
        out.alpha_min = 0.0;
        out.beta_normal_min = 0.0;
        out.beta_normal_max = 0.0;
    }
    return out;
}

template <typename T>
inline std::array<MovingPunctureBoundaryCharacteristicSummary, 6>
configure_boundary_characteristics_from_state(const BSSNGridSoA<T> &grid,
                                              const MovingPunctureEnvConfig &cfg,
                                              GaugeParameters<T> &params) {
    std::array<MovingPunctureBoundaryCharacteristicSummary, 6> summaries{};
    double gauge_speed = 1.0;

    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const bool outer = (side == 1);
            auto &summary = summaries[2 * axis + side];
            summary = summarize_boundary_face_characteristics(grid, cfg, axis, outer);
            if (summary.active && summary.radiative && !summary.reflective)
                gauge_speed = std::max(gauge_speed, summary.gauge_speed_max);
        }
    }

    params.boundary_gauge_characteristic_speed = T(gauge_speed);
    params.boundary_z4c_characteristic_speed = T(1.0);
    params.boundary_khat_characteristic_speed = T(std::sqrt(2.0));

    for (auto &summary : summaries) {
        summary.configured_gauge_speed =
            (summary.active && summary.radiative && !summary.reflective) ? gauge_speed : 0.0;
        summary.mismatch = summary.active && !summary.reflective &&
                           (summary.gauge_speed_max > summary.configured_gauge_speed + 1.0e-12);
    }

    return summaries;
}

template <typename T>
inline void report_boundary_characteristics(
    const GaugeParameters<T> &params, bool fail_on_gauge_bc_mismatch,
    const std::array<MovingPunctureBoundaryCharacteristicSummary, 6> &summaries) {
    std::printf("[bc.char] gauge_bc_speed=%.6f z4c_bc_speed=%.6f khat_bc_speed=%.6f\n",
                static_cast<double>(params.boundary_gauge_characteristic_speed),
                static_cast<double>(params.boundary_z4c_characteristic_speed),
                static_cast<double>(params.boundary_khat_characteristic_speed));

    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const bool outer = (side == 1);
            const auto &summary = summaries[2 * axis + side];
            std::printf(
                "[bc.char] face=%s active=%d radiative=%d reflective=%d alpha=[%.6e, %.6e] "
                "beta_n=[%.6e, %.6e] gauge_speed_max=%.6f bc_gauge_speed=%.6f samples=%zu\n",
                boundary_face_name(axis, outer), summary.active ? 1 : 0,
                summary.radiative ? 1 : 0, summary.reflective ? 1 : 0, summary.alpha_min,
                summary.alpha_max, summary.beta_normal_min, summary.beta_normal_max,
                summary.gauge_speed_max, summary.configured_gauge_speed, summary.samples);

            if (summary.mismatch) {
                std::ostringstream oss;
                oss << "Face " << boundary_face_name(axis, outer)
                    << " has no radiative gauge boundary speed budget for an estimated gauge "
                       "speed "
                    << summary.gauge_speed_max
                    << ". This face is likely to reflect or admit gauge content.";
                if (fail_on_gauge_bc_mismatch)
                    throw std::runtime_error(oss.str());
                std::fprintf(stderr, "[bc][warn] %s\n", oss.str().c_str());
            }
        }
    }
}

} // namespace tensorium_RG::bssn
