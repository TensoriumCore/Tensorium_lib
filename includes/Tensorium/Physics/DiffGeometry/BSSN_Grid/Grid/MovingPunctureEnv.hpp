#pragma once

#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../FMR/BSSNFixedMeshRefinement.hpp"
#include "../InitialData/BSSNInitialData.hpp"
#include "BSSNGridOperations.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

enum class MovingPunctureFMRLayout {
    Nested,
    BBHSplit,
};

struct MovingPunctureFMRConfig {
    bool   enabled = true;
    size_t levels = 2;
    size_t refinement_ratio = 2;
    size_t halo_cells = 0;
    bool   fine_levels_use_parent_init = true;
    bool   move_with_punctures = false;
    size_t regrid_interval = 0;
    double regrid_threshold_cells = 2.0;
    bool   tracker_recenter_on_drift = true;
    double tracker_recenter_cells = 4.0;
    double tracker_drift_warn_cells = 1.5;
    double outer_box_half_width = 0.0;
    double finest_box_half_width = 0.0;
    double puncture_buffer = 0.0;
    double init_core_half_width = 0.0;
    double init_transition_width = 0.0;
    MovingPunctureFMRLayout layout = MovingPunctureFMRLayout::Nested;
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
    bool   tp_calculate_target_masses = false;
    double tp_adm_tol = 1.0e-10;
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
    double                  ko_boundary_boost = 0.0;
    double                  ko_edge_corner_boost = 0.0;
    bool                    allow_reflective_bc = false;
    bool                    fail_on_gauge_bc_mismatch = false;
    MovingPunctureFMRConfig fmr{};
    tensorium::backend::Options backend_options{};
    bool                        backend_auto = true;
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

inline std::string ascii_lower(std::string value) {
    for (char &ch : value)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return value;
}

inline std::optional<std::string> env_string_lower(const char *name) {
    if (const char *raw = std::getenv(name))
        return ascii_lower(raw);
    return std::nullopt;
}

} // namespace detail

inline const char *moving_puncture_fmr_layout_name(MovingPunctureFMRLayout layout) {
    switch (layout) {
    case MovingPunctureFMRLayout::BBHSplit:
        return "bbh_split";
    case MovingPunctureFMRLayout::Nested:
    default:
        return "nested";
    }
}

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
    if (const char *raw_backend = std::getenv("TENSORIUM_MOVING_PUNCTURE_BACKEND")) {
        const std::string backend = detail::ascii_lower(raw_backend);
        if (backend.empty() || backend == "auto") {
            cfg.backend_auto = true;
        } else if (backend == "cpu") {
            cfg.backend_auto = false;
            cfg.backend_options.backend = tensorium::backend::Kind::CPU;
        } else if (backend == "cuda" || backend == "gpu") {
            cfg.backend_auto = false;
            cfg.backend_options.backend = tensorium::backend::Kind::CUDA;
        } else {
            throw std::invalid_argument(
                "TENSORIUM_MOVING_PUNCTURE_BACKEND must be one of: auto, cpu, cuda");
        }
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_CUDA_DEVICE")) {
        if (*parsed >= 0)
            cfg.backend_options.device_ordinal = static_cast<int>(*parsed);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_DEVICE_ORDINAL")) {
        if (*parsed >= 0)
            cfg.backend_options.device_ordinal = static_cast<int>(*parsed);
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
    cfg.tp_calculate_target_masses = detail::env_bool_or(
        "TENSORIUM_MOVING_PUNCTURE_TP_CALCULATE_TARGET_MASSES", cfg.use_interpolated_init);
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_TP_ADM_TOL")) {
        if (*parsed > 0.0)
            cfg.tp_adm_tol = *parsed;
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
    params.apply_rhs_sommerfeld = false;
    params.apply_rhs_sommerfeld = detail::env_bool_or(
        "TENSORIUM_MOVING_PUNCTURE_APPLY_RHS_SOMMERFELD", params.apply_rhs_sommerfeld);

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
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KO_BOUNDARY_BOOST")) {
        cfg.ko_boundary_boost = std::max(*parsed, 0.0);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_KO_EDGE_CORNER_BOOST")) {
        cfg.ko_edge_corner_boost = std::max(*parsed, 0.0);
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

    cfg.fmr.enabled = detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_ENABLE_FMR", cfg.fmr.enabled);
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_FMR_LEVELS")) {
        if (*parsed >= 0)
            cfg.fmr.levels = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_FMR_REFINEMENT_RATIO")) {
        if (*parsed >= 2)
            cfg.fmr.refinement_ratio = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_FMR_HALO_CELLS")) {
        if (*parsed >= 0)
            cfg.fmr.halo_cells = static_cast<size_t>(*parsed);
    }
    cfg.fmr.fine_levels_use_parent_init =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_FMR_FINE_LEVELS_USE_PARENT_INIT",
                            cfg.fmr.fine_levels_use_parent_init);
    cfg.fmr.move_with_punctures =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_FMR_MOVE_WITH_PUNCTURES",
                            cfg.fmr.move_with_punctures);
    if (const auto parsed = detail::env_long("TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_INTERVAL")) {
        if (*parsed >= 0)
            cfg.fmr.regrid_interval = static_cast<size_t>(*parsed);
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_THRESHOLD_CELLS")) {
        if (*parsed > 0.0)
            cfg.fmr.regrid_threshold_cells = *parsed;
    }
    cfg.fmr.tracker_recenter_on_drift =
        detail::env_bool_or("TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_ON_DRIFT",
                            cfg.fmr.tracker_recenter_on_drift);
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_CELLS")) {
        if (*parsed > 0.0)
            cfg.fmr.tracker_recenter_cells = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_TRACKER_DRIFT_WARN_CELLS")) {
        if (*parsed > 0.0)
            cfg.fmr.tracker_drift_warn_cells = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_OUTER_BOX_HALF_WIDTH")) {
        if (*parsed > 0.0)
            cfg.fmr.outer_box_half_width = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_FINEST_BOX_HALF_WIDTH")) {
        if (*parsed > 0.0)
            cfg.fmr.finest_box_half_width = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_PUNCTURE_BUFFER")) {
        if (*parsed > 0.0)
            cfg.fmr.puncture_buffer = *parsed;
    }
    if (const auto parsed = detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_INIT_CORE_HALF_WIDTH")) {
        if (*parsed > 0.0)
            cfg.fmr.init_core_half_width = *parsed;
    }
    if (const auto parsed =
            detail::env_double("TENSORIUM_MOVING_PUNCTURE_FMR_INIT_TRANSITION_WIDTH")) {
        if (*parsed > 0.0)
            cfg.fmr.init_transition_width = *parsed;
    }
    if (const auto parsed = detail::env_string_lower("TENSORIUM_MOVING_PUNCTURE_FMR_LAYOUT")) {
        if (*parsed == "nested" || *parsed == "chain")
            cfg.fmr.layout = MovingPunctureFMRLayout::Nested;
        else if (*parsed == "bbh_split" || *parsed == "bbh" || *parsed == "split")
            cfg.fmr.layout = MovingPunctureFMRLayout::BBHSplit;
        else
            throw std::runtime_error(
                "Unknown moving-puncture FMR layout. Supported values: nested, bbh_split");
    }
    if (cfg.fmr.levels == 0)
        cfg.fmr.enabled = false;

    return cfg;
}

namespace detail {

template <typename T>
inline fmr::PatchBox moving_puncture_axis_aligned_patch_from_half_width_centered(
    const BSSNGridSoA<T> &grid, const std::array<double, 3> &center, double half_width,
    size_t clearance) {
    auto clamp_axis = [&](double center_coord, double origin, double spacing, size_t cells,
                          size_t &a0, size_t &a1) {
        long lo =
            static_cast<long>(std::ceil(((center_coord - half_width) - origin) / spacing - 1.0e-12));
        long hi =
            static_cast<long>(std::floor(((center_coord + half_width) - origin) / spacing + 1.0e-12));
        if (hi < lo)
            throw std::runtime_error("Requested moving-puncture FMR box does not fit on the grid");

        const long count = hi - lo + 1;
        const long min_lo = static_cast<long>(clearance);
        const long max_hi = static_cast<long>(cells) - static_cast<long>(clearance) - 1;
        if (count <= 0 || count + 2 * static_cast<long>(clearance) > static_cast<long>(cells))
            throw std::runtime_error("Requested moving-puncture FMR box does not fit on the grid");

        if (lo < min_lo) {
            const long shift = min_lo - lo;
            lo += shift;
            hi += shift;
        }
        if (hi > max_hi) {
            const long shift = hi - max_hi;
            lo -= shift;
            hi -= shift;
        }
        if (lo < min_lo || hi > max_hi || hi < lo)
            throw std::runtime_error("Requested moving-puncture FMR box does not fit on the grid");

        a0 = static_cast<size_t>(lo);
        a1 = static_cast<size_t>(hi + 1);
    };

    fmr::PatchBox box{};
    clamp_axis(center[0], static_cast<double>(grid.x0), static_cast<double>(grid.dx), grid.dims.nx,
               box.i0, box.i1);
    clamp_axis(center[1], static_cast<double>(grid.y0), static_cast<double>(grid.dy), grid.dims.ny,
               box.j0, box.j1);
    clamp_axis(center[2], static_cast<double>(grid.z0), static_cast<double>(grid.dz), grid.dims.nz,
               box.k0, box.k1);
    return box;
}

template <typename T>
inline std::pair<fmr::PatchBox, fmr::PatchBox> moving_puncture_disjoint_split_leaf_boxes(
    const BSSNGridSoA<T> &grid, const std::array<double, 3> &puncture1,
    const std::array<double, 3> &puncture2, double half_width, size_t clearance) {
    auto left_box =
        moving_puncture_axis_aligned_patch_from_half_width_centered(grid, puncture1, half_width, clearance);
    auto right_box =
        moving_puncture_axis_aligned_patch_from_half_width_centered(grid, puncture2, half_width, clearance);
    if (!fmr::detail::patch_boxes_overlap(left_box, right_box))
        return {left_box, right_box};

    const auto coord_to_cell = [](double coord, double origin, double spacing, size_t cells) -> size_t {
        long idx = static_cast<long>(std::llround((coord - origin) / spacing));
        idx = std::clamp(idx, 0L, static_cast<long>(cells) - 1L);
        return static_cast<size_t>(idx);
    };

    const std::array<size_t, 3> puncture1_cells{
        coord_to_cell(puncture1[0], static_cast<double>(grid.x0), static_cast<double>(grid.dx), grid.dims.nx),
        coord_to_cell(puncture1[1], static_cast<double>(grid.y0), static_cast<double>(grid.dy), grid.dims.ny),
        coord_to_cell(puncture1[2], static_cast<double>(grid.z0), static_cast<double>(grid.dz), grid.dims.nz),
    };
    const std::array<size_t, 3> puncture2_cells{
        coord_to_cell(puncture2[0], static_cast<double>(grid.x0), static_cast<double>(grid.dx), grid.dims.nx),
        coord_to_cell(puncture2[1], static_cast<double>(grid.y0), static_cast<double>(grid.dy), grid.dims.ny),
        coord_to_cell(puncture2[2], static_cast<double>(grid.z0), static_cast<double>(grid.dz), grid.dims.nz),
    };

    int    split_axis = 0;
    size_t max_cell_separation = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const size_t sep =
            (puncture1_cells[axis] > puncture2_cells[axis]) ? (puncture1_cells[axis] - puncture2_cells[axis])
                                                            : (puncture2_cells[axis] - puncture1_cells[axis]);
        if (sep > max_cell_separation) {
            max_cell_separation = sep;
            split_axis = axis;
        }
    }
    if (max_cell_separation == 0) {
        throw std::runtime_error(
            "Computed moving-puncture BBH split finest leaves overlap and cannot be separated on the shared parent level");
    }

    const bool puncture1_first = puncture1_cells[split_axis] <= puncture2_cells[split_axis];
    auto      &first_box = puncture1_first ? left_box : right_box;
    auto      &second_box = puncture1_first ? right_box : left_box;
    const size_t first_cell = puncture1_first ? puncture1_cells[split_axis] : puncture2_cells[split_axis];
    const size_t second_cell = puncture1_first ? puncture2_cells[split_axis] : puncture1_cells[split_axis];

    if (first_cell >= second_cell) {
        throw std::runtime_error(
            "Computed moving-puncture BBH split finest leaves overlap and cannot retain ordered split boxes");
    }

    const size_t split_index = first_cell + (second_cell - first_cell + 1) / 2;
    auto         axis_bounds = [&](fmr::PatchBox &box) -> std::pair<size_t &, size_t &> {
        if (split_axis == 0)
            return {box.i0, box.i1};
        if (split_axis == 1)
            return {box.j0, box.j1};
        return {box.k0, box.k1};
    };

    {
        auto [first_lo, first_hi] = axis_bounds(first_box);
        auto [second_lo, second_hi] = axis_bounds(second_box);
        first_hi = std::min(first_hi, split_index);
        second_lo = std::max(second_lo, split_index);

        if (!(first_lo <= first_cell && first_cell < first_hi && second_lo <= second_cell &&
              second_cell < second_hi)) {
            throw std::runtime_error(
                "Computed moving-puncture BBH split finest leaves overlap and adaptive clipping would exclude a puncture");
        }
    }

    if (fmr::detail::patch_boxes_overlap(left_box, right_box)) {
        throw std::runtime_error(
            "Computed moving-puncture BBH split finest leaves overlap even after adaptive split clipping");
    }

    return {left_box, right_box};
}

inline fmr::PatchBox centered_patch_from_counts(size_t parent_nx, size_t parent_ny, size_t parent_nz,
                                                size_t patch_nx, size_t patch_ny, size_t patch_nz,
                                                size_t clearance) {
    auto make_axis = [&](size_t cells, size_t patch_cells, size_t &a0, size_t &a1) {
        if (patch_cells + 2 * clearance > cells)
            throw std::runtime_error(
                "Requested moving-puncture FMR level exceeds the parent grid extent");
        a0 = (cells - patch_cells) / 2;
        a1 = a0 + patch_cells;
        if (a0 < clearance || a1 + clearance > cells)
            throw std::runtime_error(
                "Requested moving-puncture FMR level leaves insufficient boundary clearance");
    };

    fmr::PatchBox box{};
    make_axis(parent_nx, patch_nx, box.i0, box.i1);
    make_axis(parent_ny, patch_ny, box.j0, box.j1);
    make_axis(parent_nz, patch_nz, box.k0, box.k1);
    return box;
}

template <typename T>
inline double moving_puncture_axis_half_extent(T origin, T spacing, size_t cells) {
    const double x_min = std::abs(static_cast<double>(origin));
    const double x_max = std::abs(static_cast<double>(origin + T(cells - 1) * spacing));
    return std::min(x_min, x_max);
}

} // namespace detail

template <typename T>
inline std::vector<fmr::LevelConfig>
build_moving_puncture_fmr_levels(const BSSNGridSoA<T> &root_grid,
                                 const MovingPunctureEnvConfig &cfg,
                                 const std::array<double, 3> &center = {0.0, 0.0, 0.0}) {
    std::vector<fmr::LevelConfig> levels;
    if (!cfg.fmr.enabled || cfg.fmr.levels == 0)
        return levels;
    if (cfg.fmr.refinement_ratio < 2)
        throw std::invalid_argument("Moving-puncture FMR refinement ratio must be >= 2");

    constexpr size_t clearance = 2;
    const double auto_buffer =
        (cfg.fmr.puncture_buffer > 0.0) ? cfg.fmr.puncture_buffer : std::max(2.0, 4.0 * cfg.spacing);
    const double scale =
        std::pow(static_cast<double>(cfg.fmr.refinement_ratio), static_cast<double>(cfg.fmr.levels - 1));
    const double max_root_half_width = std::min(
        {detail::moving_puncture_axis_half_extent(root_grid.x0, root_grid.dx, root_grid.dims.nx) -
             2.0 * static_cast<double>(root_grid.dx),
         detail::moving_puncture_axis_half_extent(root_grid.y0, root_grid.dy, root_grid.dims.ny) -
             2.0 * static_cast<double>(root_grid.dy),
         detail::moving_puncture_axis_half_extent(root_grid.z0, root_grid.dz, root_grid.dims.nz) -
             2.0 * static_cast<double>(root_grid.dz)});
    if (!(max_root_half_width > 0.0))
        throw std::runtime_error("Moving-puncture FMR requires positive room inside the root grid");

    double outer_half_width = 0.0;
    double finest_half_width = 0.0;
    if (cfg.fmr.outer_box_half_width > 0.0) {
        outer_half_width = cfg.fmr.outer_box_half_width;
        if (outer_half_width > max_root_half_width) {
            throw std::runtime_error(
                "Requested moving-puncture outer FMR box is too large for the root domain");
        }
        finest_half_width = outer_half_width / std::max(1.0, scale);
    } else {
        // Keep automatic patches no larger than their parent by default; otherwise
        // "refinement" can cost more than a full-domain fine solve on the first step.
        const double auto_outer_half_width_limit =
            max_root_half_width / static_cast<double>(cfg.fmr.refinement_ratio);
        const double auto_finest_half_width_limit =
            auto_outer_half_width_limit / std::max(1.0, scale);
        const double min_binary_half_width = std::max(cfg.separation, 4.0 * cfg.spacing);
        if (auto_finest_half_width_limit < min_binary_half_width) {
            throw std::runtime_error(
                "Automatic moving-puncture FMR boxes cannot cover the binary without oversized "
                "refined levels; reduce FMR levels or set explicit box widths");
        }
        const double auto_finest_half_width = std::max(cfg.separation + auto_buffer, 4.0 * cfg.spacing);
        finest_half_width =
            (cfg.fmr.finest_box_half_width > 0.0) ? cfg.fmr.finest_box_half_width : auto_finest_half_width;
        const double max_finest_half_width = max_root_half_width / std::max(1.0, scale);
        if (cfg.fmr.finest_box_half_width > 0.0 && finest_half_width > max_finest_half_width) {
            throw std::runtime_error(
                "Requested moving-puncture finest FMR box is too large for the root domain");
        }
        if (cfg.fmr.finest_box_half_width == 0.0)
            finest_half_width = std::min(finest_half_width, auto_finest_half_width_limit);
        finest_half_width = std::min(finest_half_width, max_finest_half_width);
        outer_half_width = finest_half_width * scale;
    }
    if (!(finest_half_width > 0.0))
        throw std::runtime_error("Computed moving-puncture finest FMR box is empty");

    levels.reserve(cfg.fmr.levels);
    const BSSNGridSoA<T> *parent_grid = &root_grid;
    std::vector<std::unique_ptr<BSSNGridSoA<T>>> temporary_parents;
    double level_half_width = outer_half_width;

    for (size_t level = 0; level < cfg.fmr.levels; ++level) {
        const fmr::PatchBox patch = detail::moving_puncture_axis_aligned_patch_from_half_width_centered(
            *parent_grid, center, level_half_width, clearance);
        levels.push_back({patch, cfg.fmr.refinement_ratio, cfg.fmr.halo_cells});

        if (level + 1 >= cfg.fmr.levels)
            break;

        const auto child = fmr::detail::make_child_geometry(*parent_grid, levels.back());
        auto next_parent = std::make_unique<BSSNGridSoA<T>>(child.nx, child.ny, child.nz, child.ng,
                                                            T(child.dx), T(child.dy), T(child.dz));
        next_parent->x0 = T(child.x0);
        next_parent->y0 = T(child.y0);
        next_parent->z0 = T(child.z0);
        parent_grid = next_parent.get();
        temporary_parents.push_back(std::move(next_parent));
        level_half_width /= static_cast<double>(cfg.fmr.refinement_ratio);
    }

    return levels;
}

inline std::array<std::array<double, 3>, 2>
moving_puncture_initial_puncture_positions(const MovingPunctureEnvConfig &cfg) {
    if (cfg.use_interpolated_init)
        return {{{-cfg.separation, 0.0, 0.0}, {cfg.separation, 0.0, 0.0}}};
    return {{{0.0, cfg.separation, 0.0}, {0.0, -cfg.separation, 0.0}}};
}

template <typename T>
inline fmr::BinaryPunctureHierarchyConfig
build_moving_puncture_bbh_split_hierarchy_config(
    const BSSNGridSoA<T> &root_grid, const MovingPunctureEnvConfig &cfg,
    const std::array<double, 3> &puncture1, const std::array<double, 3> &puncture2) {
    if (!cfg.fmr.enabled || cfg.fmr.levels == 0)
        return {};
    if (cfg.fmr.levels < 2) {
        throw std::invalid_argument(
            "BBH split FMR requires at least two refined levels: one shared level and split leaves");
    }
    if (cfg.fmr.refinement_ratio < 2)
        throw std::invalid_argument("Moving-puncture BBH split refinement ratio must be >= 2");

    constexpr size_t clearance = 2;
    const double auto_buffer =
        (cfg.fmr.puncture_buffer > 0.0) ? cfg.fmr.puncture_buffer : std::max(2.0, 4.0 * cfg.spacing);
    const double leaf_half_width =
        (cfg.fmr.finest_box_half_width > 0.0) ? cfg.fmr.finest_box_half_width
                                              : std::max(auto_buffer, 4.0 * cfg.spacing);

    const std::array<double, 3> common_center{
        0.5 * (puncture1[0] + puncture2[0]), 0.5 * (puncture1[1] + puncture2[1]),
        0.5 * (puncture1[2] + puncture2[2])};
    const auto puncture_extent_from_center = [&](int axis) {
        return std::max(std::abs(puncture1[axis] - common_center[axis]),
                        std::abs(puncture2[axis] - common_center[axis]));
    };
    const double split_parent_half_width =
        std::max({puncture_extent_from_center(0), puncture_extent_from_center(1),
                  puncture_extent_from_center(2)}) +
        leaf_half_width;

    const size_t shared_levels = cfg.fmr.levels - 1;
    const double shared_scale = std::pow(static_cast<double>(cfg.fmr.refinement_ratio),
                                         static_cast<double>(shared_levels - 1));
    const double max_root_half_width = std::min(
        {detail::moving_puncture_axis_half_extent(root_grid.x0, root_grid.dx, root_grid.dims.nx) -
             2.0 * static_cast<double>(root_grid.dx),
         detail::moving_puncture_axis_half_extent(root_grid.y0, root_grid.dy, root_grid.dims.ny) -
             2.0 * static_cast<double>(root_grid.dy),
         detail::moving_puncture_axis_half_extent(root_grid.z0, root_grid.dz, root_grid.dims.nz) -
             2.0 * static_cast<double>(root_grid.dz)});
    if (!(max_root_half_width > 0.0))
        throw std::runtime_error("Moving-puncture BBH split FMR requires positive room inside the root grid");

    double outer_half_width = 0.0;
    if (cfg.fmr.outer_box_half_width > 0.0) {
        outer_half_width = cfg.fmr.outer_box_half_width;
        if (outer_half_width > max_root_half_width) {
            throw std::runtime_error(
                "Requested moving-puncture BBH split outer FMR box is too large for the root domain");
        }
        const double last_shared_half_width = outer_half_width / std::max(1.0, shared_scale);
        if (last_shared_half_width + 1.0e-12 < split_parent_half_width) {
            throw std::runtime_error(
                "Requested moving-puncture BBH split outer box is too small to contain both puncture leaves");
        }
    } else {
        outer_half_width = split_parent_half_width * shared_scale;
        if (outer_half_width > max_root_half_width) {
            throw std::runtime_error(
                "Automatic moving-puncture BBH split boxes exceed the root domain; reduce levels or widths");
        }
    }

    fmr::BinaryPunctureHierarchyConfig out;
    out.shared_levels.reserve(shared_levels);

    const BSSNGridSoA<T> *parent_grid = &root_grid;
    std::vector<std::unique_ptr<BSSNGridSoA<T>>> temporary_parents;
    for (size_t level = 0; level < shared_levels; ++level) {
        const double level_half_width =
            outer_half_width / std::pow(static_cast<double>(cfg.fmr.refinement_ratio),
                                        static_cast<double>(level));
        const auto patch = detail::moving_puncture_axis_aligned_patch_from_half_width_centered(
            *parent_grid, common_center, level_half_width, clearance);
        out.shared_levels.push_back({patch, cfg.fmr.refinement_ratio, cfg.fmr.halo_cells});

        const auto child = fmr::detail::make_child_geometry(*parent_grid, out.shared_levels.back());
        auto next_parent = std::make_unique<BSSNGridSoA<T>>(child.nx, child.ny, child.nz, child.ng,
                                                            T(child.dx), T(child.dy), T(child.dz));
        next_parent->x0 = T(child.x0);
        next_parent->y0 = T(child.y0);
        next_parent->z0 = T(child.z0);
        parent_grid = next_parent.get();
        temporary_parents.push_back(std::move(next_parent));
    }

    const auto [left_leaf_box, right_leaf_box] = detail::moving_puncture_disjoint_split_leaf_boxes(
        *parent_grid, puncture1, puncture2, leaf_half_width, clearance);
    out.left_leaf = {left_leaf_box, cfg.fmr.refinement_ratio, cfg.fmr.halo_cells};
    out.right_leaf = {right_leaf_box, cfg.fmr.refinement_ratio, cfg.fmr.halo_cells};

    if (fmr::detail::patch_boxes_overlap(out.left_leaf.parent_cells, out.right_leaf.parent_cells)) {
        throw std::runtime_error(
            "Computed moving-puncture BBH split finest leaves overlap; increase outer width or decrease finest width");
    }

    return out;
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
            1e-10, cfg.tp_calculate_target_masses, cfg.tp_adm_tol);
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
    configure_ko_boundary_taper(cfg.ko_boundary_width, cfg.ko_boundary_floor,
                                cfg.ko_boundary_boost, cfg.ko_edge_corner_boost);
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
