#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tensorium_RG::bssn {

struct MovingPunctureEccentricityControlConfig {
    bool   enabled = false;
    bool   tune_only = false;
    bool   export_trials = false;
    size_t iterations = 2;
    size_t trial_steps = 800;
    double fit_tmin = 0.0;
    double fit_tmax = 0.0;
    double fit_start_fraction = 0.15;
    double fit_end_fraction = 1.0;
    size_t min_samples = 48;
    double tangential_gain = 1.0;
    double radial_gain = 1.0;
    double max_fractional_tangential_update = 0.15;
    double max_absolute_radial_update = 1.0e-3;
};

struct MovingPunctureTrackPoint {
    double t = std::numeric_limits<double>::quiet_NaN();
    bool   has_left = false;
    bool   has_right = false;
    double x_left = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double x_right = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
};

struct MovingPunctureEccentricityFitResult {
    bool   valid = false;
    size_t used_samples = 0;
    double reference_time = 0.0;
    double fit_tmin = 0.0;
    double fit_tmax = 0.0;
    double initial_separation = 0.0;
    double mean_separation = 0.0;
    double secular_rdot = 0.0;
    double orbital_frequency = 0.0;
    double phase_advance = 0.0;
    double cos_coefficient = 0.0;
    double sin_coefficient = 0.0;
    double oscillation_amplitude = 0.0;
    double eccentricity = 0.0;
    double oscillatory_rdot0 = 0.0;
    double fit_rmse = 0.0;
};

struct MovingPunctureEccentricityUpdate {
    bool   valid = false;
    double reduced_mass = 0.0;
    double delta_tangential_momentum = 0.0;
    double delta_radial_momentum = 0.0;
    double new_tangential_momentum = 0.0;
    double new_radial_momentum = 0.0;
};

namespace detail {

inline bool solve_augmented_4x4(std::array<std::array<double, 5>, 4> &augmented) {
    constexpr int n = 4;
    for (int pivot = 0; pivot < n; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < n; ++row) {
            if (std::abs(augmented[row][pivot]) > std::abs(augmented[best][pivot]))
                best = row;
        }
        if (std::abs(augmented[best][pivot]) < 1.0e-14)
            return false;
        if (best != pivot)
            std::swap(augmented[best], augmented[pivot]);

        const double inv_pivot = 1.0 / augmented[pivot][pivot];
        for (int col = pivot; col <= n; ++col)
            augmented[pivot][col] *= inv_pivot;

        for (int row = 0; row < n; ++row) {
            if (row == pivot)
                continue;
            const double factor = augmented[row][pivot];
            if (factor == 0.0)
                continue;
            for (int col = pivot; col <= n; ++col)
                augmented[row][col] -= factor * augmented[pivot][col];
        }
    }
    return true;
}

inline double unwrap_phase(double previous, double current) {
    double delta = current - previous;
    constexpr double two_pi = 2.0 * 3.141592653589793238462643383279502884;
    while (delta > 3.141592653589793238462643383279502884)
        delta -= two_pi;
    while (delta < -3.141592653589793238462643383279502884)
        delta += two_pi;
    return previous + delta;
}

struct TrackSample {
    double t = 0.0;
    double separation = 0.0;
    double phase = 0.0;
};

} // namespace detail

inline MovingPunctureEccentricityFitResult fit_moving_puncture_eccentricity(
    const std::vector<MovingPunctureTrackPoint> &track,
    const MovingPunctureEccentricityControlConfig &cfg) {
    MovingPunctureEccentricityFitResult result{};

    std::vector<detail::TrackSample> samples;
    samples.reserve(track.size());
    double previous_phase = 0.0;
    bool   have_previous_phase = false;
    for (const auto &point : track) {
        if (!(point.has_left && point.has_right))
            continue;
        if (!(std::isfinite(point.t) && std::isfinite(point.x_left) && std::isfinite(point.y_left) &&
              std::isfinite(point.x_right) && std::isfinite(point.y_right)))
            continue;
        const double dx = point.x_right - point.x_left;
        const double dy = point.y_right - point.y_left;
        const double separation = std::hypot(dx, dy);
        if (!(std::isfinite(separation) && separation > 0.0))
            continue;

        double phase = std::atan2(dy, dx);
        if (have_previous_phase)
            phase = detail::unwrap_phase(previous_phase, phase);
        previous_phase = phase;
        have_previous_phase = true;
        samples.push_back({point.t, separation, phase});
    }

    if (samples.size() < std::max<size_t>(cfg.min_samples, size_t(8)))
        return result;

    const double t0 = samples.front().t;
    const double t1 = samples.back().t;
    const double duration = t1 - t0;
    if (!(std::isfinite(duration) && duration > 0.0))
        return result;

    result.reference_time = t0;
    result.initial_separation = samples.front().separation;

    const double start_fraction = std::clamp(cfg.fit_start_fraction, 0.0, 0.95);
    const double end_fraction = std::clamp(cfg.fit_end_fraction, start_fraction + 0.01, 1.0);
    const double default_tmin = t0 + start_fraction * duration;
    const double default_tmax = t0 + end_fraction * duration;

    result.fit_tmin = (cfg.fit_tmin > t0) ? cfg.fit_tmin : default_tmin;
    result.fit_tmax = (cfg.fit_tmax > result.fit_tmin) ? std::min(cfg.fit_tmax, t1) : default_tmax;
    if (!(result.fit_tmax > result.fit_tmin))
        return result;

    std::vector<detail::TrackSample> window;
    window.reserve(samples.size());
    for (const auto &sample : samples) {
        if (sample.t >= result.fit_tmin && sample.t <= result.fit_tmax)
            window.push_back(sample);
    }
    if (window.size() < std::max<size_t>(cfg.min_samples, size_t(8)))
        return result;

    result.used_samples = window.size();
    const double fit_duration = window.back().t - window.front().t;
    if (!(std::isfinite(fit_duration) && fit_duration > 0.0))
        return result;

    result.phase_advance = window.back().phase - window.front().phase;
    result.orbital_frequency = std::abs(result.phase_advance / fit_duration);
    if (!(std::isfinite(result.orbital_frequency) && result.orbital_frequency > 1.0e-8))
        return result;
    if (std::abs(result.phase_advance) < 0.05)
        return result;

    std::array<std::array<double, 5>, 4> augmented{};
    for (const auto &sample : window) {
        const double tau = sample.t - result.reference_time;
        const double arg = result.orbital_frequency * tau;
        const double basis[4] = {1.0, tau, std::cos(arg), std::sin(arg)};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col)
                augmented[row][col] += basis[row] * basis[col];
            augmented[row][4] += basis[row] * sample.separation;
        }
    }

    if (!detail::solve_augmented_4x4(augmented))
        return result;

    const double c0 = augmented[0][4];
    const double c1 = augmented[1][4];
    const double a = augmented[2][4];
    const double b = augmented[3][4];
    if (!(std::isfinite(c0) && std::isfinite(c1) && std::isfinite(a) && std::isfinite(b)))
        return result;
    if (std::abs(c0) < 1.0e-12)
        return result;

    double rss = 0.0;
    for (const auto &sample : window) {
        const double tau = sample.t - result.reference_time;
        const double model = c0 + c1 * tau +
                             a * std::cos(result.orbital_frequency * tau) +
                             b * std::sin(result.orbital_frequency * tau);
        const double residual = sample.separation - model;
        rss += residual * residual;
    }

    result.mean_separation = c0;
    result.secular_rdot = c1;
    result.cos_coefficient = a;
    result.sin_coefficient = b;
    result.oscillation_amplitude = std::hypot(a, b);
    result.eccentricity = result.oscillation_amplitude / std::abs(c0);
    result.oscillatory_rdot0 = result.orbital_frequency * b;
    result.fit_rmse = std::sqrt(rss / static_cast<double>(window.size()));
    result.valid = std::isfinite(result.eccentricity) && std::isfinite(result.fit_rmse);
    return result;
}

inline MovingPunctureEccentricityUpdate suggest_moving_puncture_momentum_update(
    const MovingPunctureEccentricityFitResult &fit,
    double mass1, double mass2, double tangential_momentum, double radial_momentum,
    const MovingPunctureEccentricityControlConfig &cfg) {
    MovingPunctureEccentricityUpdate update{};
    if (!fit.valid)
        return update;
    if (!(std::isfinite(mass1) && std::isfinite(mass2) && mass1 > 0.0 && mass2 > 0.0))
        return update;
    if (!(std::isfinite(tangential_momentum) && std::isfinite(radial_momentum) &&
          tangential_momentum >= 0.0 && radial_momentum >= 0.0))
        return update;

    const double total_mass = mass1 + mass2;
    const double reduced_mass = (mass1 * mass2) / total_mass;
    if (!(std::isfinite(reduced_mass) && reduced_mass > 0.0))
        return update;

    const double tangential_fraction =
        0.5 * cfg.tangential_gain * (fit.cos_coefficient / fit.mean_separation);
    const double clamped_tangential_fraction =
        std::clamp(tangential_fraction, -cfg.max_fractional_tangential_update,
                   cfg.max_fractional_tangential_update);
    update.delta_tangential_momentum = tangential_momentum * clamped_tangential_fraction;

    const double target_rdot_correction = fit.secular_rdot + fit.oscillatory_rdot0;
    const double unclamped_delta_radial =
        cfg.radial_gain * reduced_mass * target_rdot_correction;
    update.delta_radial_momentum =
        std::clamp(unclamped_delta_radial, -cfg.max_absolute_radial_update,
                   cfg.max_absolute_radial_update);

    update.reduced_mass = reduced_mass;
    update.new_tangential_momentum =
        std::max(0.0, tangential_momentum + update.delta_tangential_momentum);
    update.new_radial_momentum = std::max(0.0, radial_momentum + update.delta_radial_momentum);
    update.valid = std::isfinite(update.new_tangential_momentum) &&
                   std::isfinite(update.new_radial_momentum);
    return update;
}

} // namespace tensorium_RG::bssn
