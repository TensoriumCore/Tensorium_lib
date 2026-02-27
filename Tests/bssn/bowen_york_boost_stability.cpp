#include "../framework/TestRegistry.hpp"
#include "stability_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

namespace {

void export_slice_csv(const tensorium::tests::Grid &grid, size_t step,
                      const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/slice_" << std::setw(4) << std::setfill('0') << step << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,alpha,W,mask\n";

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;

            const size_t idx = grid.alpha.idx(i, j, k);

            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            const double W = std::sqrt(std::max(chi, 1e-16));

            const double mask = (alpha < 0.1) ? 1.0 : 0.0;

            file << x << "," << y << "," << alpha << "," << W << "," << mask << "\n";
        }
    }
}

void export_constraint_slice_csv(
    const tensorium::tests::Grid &grid,
    const tensorium::tests::ConstraintScratch &constraint_scratch, size_t step,
    const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/constraint_slice_" << std::setw(4) << std::setfill('0') << step
       << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,alpha,W,H,absH,Mx,My,Mz,Mnorm,Cx,Cy,Cz,Cnorm,Theta,Zx,Zy,Zz,Znorm\n";

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;

            const size_t idx = grid.alpha.idx(i, j, k);
            const size_t cidx = constraint_scratch.H.idx(i, j, k);

            const double alpha = double(grid.alpha.ptr()[idx]);
            const double chi = double(grid.chi.ptr()[idx]);
            const double W = std::sqrt(std::max(chi, 1e-16));

            const double H = double(constraint_scratch.H.ptr()[cidx]);
            const double Mx = double(constraint_scratch.M[0].ptr()[cidx]);
            const double My = double(constraint_scratch.M[1].ptr()[cidx]);
            const double Mz = double(constraint_scratch.M[2].ptr()[cidx]);
            const double Mnorm = std::sqrt(Mx * Mx + My * My + Mz * Mz);

            const double Cx = double(constraint_scratch.C[0].ptr()[cidx]);
            const double Cy = double(constraint_scratch.C[1].ptr()[cidx]);
            const double Cz = double(constraint_scratch.C[2].ptr()[cidx]);
            const double Cnorm = std::sqrt(Cx * Cx + Cy * Cy + Cz * Cz);

            const double Theta = double(grid.Theta.ptr()[idx]);
            const double Zx = double(grid.Z[0].ptr()[idx]);
            const double Zy = double(grid.Z[1].ptr()[idx]);
            const double Zz = double(grid.Z[2].ptr()[idx]);
            const double Znorm = std::sqrt(Zx * Zx + Zy * Zy + Zz * Zz);

            file << x << "," << y << "," << alpha << "," << W << "," << H << ","
                 << std::abs(H) << "," << Mx << "," << My << "," << Mz << "," << Mnorm << ","
                 << Cx << "," << Cy << "," << Cz << "," << Cnorm << "," << Theta << "," << Zx
                 << "," << Zy << "," << Zz << "," << Znorm << "\n";
        }
    }
}

struct PuncturePlaneSample {
    double x_left = 0.0;
    double y_left = 0.0;
    double chi_left = std::numeric_limits<double>::infinity();
    double alpha_left = std::numeric_limits<double>::infinity();
    bool   has_left = false;

    double x_right = 0.0;
    double y_right = 0.0;
    double chi_right = std::numeric_limits<double>::infinity();
    double alpha_right = std::numeric_limits<double>::infinity();
    bool   has_right = false;
};

struct PunctureTrackHistory {
    PuncturePlaneSample prev{};
    PuncturePlaneSample last{};
    double              t_prev = 0.0;
    double              t_last = 0.0;
    bool                has_prev = false;
    bool                has_last = false;
};

struct PuncturePrediction {
    double x_left = 0.0;
    double y_left = 0.0;
    double x_right = 0.0;
    double y_right = 0.0;
    bool   valid = false;
};

struct PunctureCandidate {
    size_t i = 0;
    size_t j = 0;
    double x = 0.0;
    double y = 0.0;
    double chi = std::numeric_limits<double>::infinity();
    double alpha = std::numeric_limits<double>::infinity();
    bool   found = false;
};

enum class TrackerRecenterMode {
    Mixed,
    Chi,
    Alpha,
};

const char *tracker_recenter_mode_name(TrackerRecenterMode mode) {
    switch (mode) {
    case TrackerRecenterMode::Chi:
        return "chi";
    case TrackerRecenterMode::Alpha:
        return "alpha";
    case TrackerRecenterMode::Mixed:
    default:
        return "mixed";
    }
}

std::string ascii_lower_copy(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

TrackerRecenterMode parse_tracker_recenter_mode_or(const char *name, TrackerRecenterMode fallback) {
    if (const char *raw = std::getenv(name)) {
        const std::string mode = ascii_lower_copy(std::string(raw));
        if (mode == "mixed")
            return TrackerRecenterMode::Mixed;
        if (mode == "chi")
            return TrackerRecenterMode::Chi;
        if (mode == "alpha")
            return TrackerRecenterMode::Alpha;
        std::cout << "[warn] invalid " << name << "='" << raw
                  << "' (expected mixed|chi|alpha), fallback="
                  << tracker_recenter_mode_name(fallback) << std::endl;
    }
    return fallback;
}

inline PuncturePrediction predict_puncture_positions(const PunctureTrackHistory &history,
                                                     double current_time) {
    PuncturePrediction pred;
    if (!history.has_last || !history.last.has_left || !history.last.has_right)
        return pred;

    pred.x_left = history.last.x_left;
    pred.y_left = history.last.y_left;
    pred.x_right = history.last.x_right;
    pred.y_right = history.last.y_right;

    if (history.has_prev && history.prev.has_left && history.prev.has_right) {
        const double dt_prev = history.t_last - history.t_prev;
        const double dt_now = current_time - history.t_last;
        if (dt_prev > 1e-12 && std::isfinite(dt_prev) && dt_now >= 0.0 && std::isfinite(dt_now)) {
            const double ratio = std::min(dt_now / dt_prev, 1.5);
            pred.x_left += ratio * (history.last.x_left - history.prev.x_left);
            pred.y_left += ratio * (history.last.y_left - history.prev.y_left);
            pred.x_right += ratio * (history.last.x_right - history.prev.x_right);
            pred.y_right += ratio * (history.last.y_right - history.prev.y_right);
        }
    }

    pred.valid = true;
    return pred;
}

inline bool clamp_jump_to_radius(double x_ref, double y_ref, double max_jump, double &x,
                                 double &y) {
    const double dx = x - x_ref;
    const double dy = y - y_ref;
    const double r = std::sqrt(dx * dx + dy * dy);
    if (!std::isfinite(r) || r <= max_jump || max_jump <= 0.0)
        return false;
    const double s = max_jump / r;
    x = x_ref + s * dx;
    y = y_ref + s * dy;
    return true;
}

inline double parabolic_subcell_offset(double fm, double f0, double fp) {
    const double denom = fm - 2.0 * f0 + fp;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-30)
        return 0.0;
    double offset = 0.5 * (fm - fp) / denom;
    if (!std::isfinite(offset))
        return 0.0;
    if (offset > 0.5)
        offset = 0.5;
    if (offset < -0.5)
        offset = -0.5;
    return offset;
}

PunctureCandidate find_min_candidate_in_window(const tensorium::tests::Grid &grid, size_t i_lo,
                                               size_t i_hi_exclusive, size_t j_lo,
                                               size_t j_hi_exclusive, bool use_exclusion = false,
                                               double ex_x = 0.0, double ex_y = 0.0,
                                               double ex_r2 = 0.0, bool use_anchor = false,
                                               double x_ref = 0.0, double y_ref = 0.0,
                                               double dist_weight = 0.0,
                                               double alpha_weight = 0.0,
                                               TrackerRecenterMode score_mode =
                                                   TrackerRecenterMode::Mixed) {
    PunctureCandidate out;
    double            best_score = std::numeric_limits<double>::infinity();
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;
    const double inv_dx = 1.0 / std::max(grid.dx, 1e-14);
    const double inv_dy = 1.0 / std::max(grid.dy, 1e-14);
    for (size_t i = i_lo; i < i_hi_exclusive; ++i) {
        for (size_t j = j_lo; j < j_hi_exclusive; ++j) {
            const size_t idx = grid.alpha.idx(i, j, k);
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;
            if (use_exclusion) {
                const double dx = x - ex_x;
                const double dy = y - ex_y;
                if (dx * dx + dy * dy < ex_r2)
                    continue;
            }

            const double chi = grid.chi.ptr()[idx];
            const double alpha = grid.alpha.ptr()[idx];
            if (!std::isfinite(chi) || !std::isfinite(alpha))
                continue;

            double score = chi;
            if (score_mode == TrackerRecenterMode::Alpha) {
                score = alpha;
            } else if (score_mode == TrackerRecenterMode::Mixed) {
                score = chi + alpha_weight * alpha;
            }
            if (use_anchor) {
                const double dx_cells = (x - x_ref) * inv_dx;
                const double dy_cells = (y - y_ref) * inv_dy;
                score += dist_weight * (dx_cells * dx_cells + dy_cells * dy_cells);
            }

            if (!out.found || score < best_score ||
                (score == best_score && (chi < out.chi || (chi == out.chi && alpha < out.alpha)))) {
                out.found = true;
                out.i = i;
                out.j = j;
                out.x = x;
                out.y = y;
                out.chi = chi;
                out.alpha = alpha;
                best_score = score;
            }
        }
    }
    return out;
}

void refine_candidate_xy(const tensorium::tests::Grid &grid, PunctureCandidate &cand) {
    if (!cand.found)
        return;
    const size_t ng = grid.dims.ng;
    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t k = ng + grid.dims.nz / 2;
    const size_t i_min = ng;
    const size_t i_max = ng + nx - 1;
    const size_t j_min = ng;
    const size_t j_max = ng + ny - 1;
    const size_t i = cand.i;
    const size_t j = cand.j;

    const size_t cidx = grid.alpha.idx(i, j, k);
    const double x_center = grid.x0 + (double(i) - double(ng)) * grid.dx;
    const double y_center = grid.y0 + (double(j) - double(ng)) * grid.dy;
    double       dx_sub = 0.0;
    double       dy_sub = 0.0;

    if (i > i_min && i < i_max) {
        const double chim = grid.chi.ptr()[grid.alpha.idx(i - 1, j, k)];
        const double chi0 = grid.chi.ptr()[cidx];
        const double chip = grid.chi.ptr()[grid.alpha.idx(i + 1, j, k)];
        dx_sub = parabolic_subcell_offset(chim, chi0, chip);
    }
    if (j > j_min && j < j_max) {
        const double chim = grid.chi.ptr()[grid.alpha.idx(i, j - 1, k)];
        const double chi0 = grid.chi.ptr()[cidx];
        const double chip = grid.chi.ptr()[grid.alpha.idx(i, j + 1, k)];
        dy_sub = parabolic_subcell_offset(chim, chi0, chip);
    }

    cand.x = x_center + dx_sub * grid.dx;
    cand.y = y_center + dy_sub * grid.dy;
}

size_t nearest_index(double x, double origin, double spacing, size_t lo, size_t hi_inclusive) {
    const long idx = std::lround((x - origin) / spacing + double(lo));
    const long clamped = std::max<long>(static_cast<long>(lo),
                                        std::min<long>(static_cast<long>(hi_inclusive), idx));
    return static_cast<size_t>(clamped);
}

bool parse_env_bool_or(const char *name, bool fallback) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw)
            return parsed != 0;
    }
    return fallback;
}

struct CircularMomentumSuggestion {
    bool   valid = false;
    double d = 0.0;
    double total_mass = 0.0;
    double reduced_mass = 0.0;
    double p_newtonian = 0.0;
    double p_pn = 0.0;
};

CircularMomentumSuggestion suggest_circular_momentum(double m1, double m2, double separation) {
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

struct ShiftPunctureTracker {
    std::array<double, 3> p1{0.0, 0.0, 0.0};
    std::array<double, 3> p2{0.0, 0.0, 0.0};
    std::array<double, 3> beta1_prev{0.0, 0.0, 0.0};
    std::array<double, 3> beta2_prev{0.0, 0.0, 0.0};
    bool                  initialized = false;
};

bool trilinear_indices(const tensorium::tests::Grid &grid, double x, double y, double z, size_t &i0,
                       size_t &j0, size_t &k0, double &tx, double &ty, double &tz) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    if (I1 <= I0 + 1 || J1 <= J0 + 1 || K1 <= K0 + 1)
        return false;

    auto coord_to_index = [](double c, double c0, double dc, size_t i_lo, size_t i_hi, size_t &i,
                             double &t) -> bool {
        const double q = (c - c0) / dc + double(i_lo);
        if (!std::isfinite(q))
            return false;
        double q_clamped = q;
        if (q_clamped < double(i_lo))
            q_clamped = double(i_lo);
        if (q_clamped > double(i_hi))
            q_clamped = double(i_hi);

        long il = static_cast<long>(std::floor(q_clamped));
        if (il < static_cast<long>(i_lo))
            il = static_cast<long>(i_lo);
        if (il >= static_cast<long>(i_hi)) {
            il = static_cast<long>(i_hi - 1);
            t = 1.0;
        } else {
            t = q_clamped - double(il);
        }
        i = static_cast<size_t>(il);
        return true;
    };

    if (!coord_to_index(x, grid.x0, grid.dx, I0, I1 - 1, i0, tx))
        return false;
    if (!coord_to_index(y, grid.y0, grid.dy, J0, J1 - 1, j0, ty))
        return false;
    if (!coord_to_index(z, grid.z0, grid.dz, K0, K1 - 1, k0, tz))
        return false;

    return true;
}

inline void lagrange4_weights(double t, double w[4]) {
    // 4-point Lagrange interpolation on {-1,0,1,2} around the local cell.
    // t in [0,1] is the fractional offset from the i0 node.
    w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
    w[1] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
    w[2] = -(t + 1.0) * t * (t - 2.0) / 2.0;
    w[3] = (t + 1.0) * t * (t - 1.0) / 6.0;
}

template <typename FieldLike>
bool sample_scalar_trilinear(const tensorium::tests::Grid &grid, const FieldLike &field, double x,
                             double y, double z, double &value) {
    size_t i0 = 0, j0 = 0, k0 = 0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    if (!trilinear_indices(grid, x, y, z, i0, j0, k0, tx, ty, tz))
        return false;
    const size_t i1 = i0 + 1;
    const size_t j1 = j0 + 1;
    const size_t k1 = k0 + 1;

    auto at = [&](size_t i, size_t j, size_t k) { return double(field.ptr()[field.idx(i, j, k)]); };
    const double c000 = at(i0, j0, k0);
    const double c100 = at(i1, j0, k0);
    const double c010 = at(i0, j1, k0);
    const double c110 = at(i1, j1, k0);
    const double c001 = at(i0, j0, k1);
    const double c101 = at(i1, j0, k1);
    const double c011 = at(i0, j1, k1);
    const double c111 = at(i1, j1, k1);

    if (!std::isfinite(c000) || !std::isfinite(c100) || !std::isfinite(c010) || !std::isfinite(c110) ||
        !std::isfinite(c001) || !std::isfinite(c101) || !std::isfinite(c011) || !std::isfinite(c111))
        return false;

    const double c00 = c000 * (1.0 - tx) + c100 * tx;
    const double c10 = c010 * (1.0 - tx) + c110 * tx;
    const double c01 = c001 * (1.0 - tx) + c101 * tx;
    const double c11 = c011 * (1.0 - tx) + c111 * tx;
    const double c0 = c00 * (1.0 - ty) + c10 * ty;
    const double c1 = c01 * (1.0 - ty) + c11 * ty;
    value = c0 * (1.0 - tz) + c1 * tz;
    return std::isfinite(value);
}

template <typename FieldLike>
bool sample_scalar_lagrange4(const tensorium::tests::Grid &grid, const FieldLike &field, double x,
                             double y, double z, double &value) {
    size_t i0 = 0, j0 = 0, k0 = 0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    if (!trilinear_indices(grid, x, y, z, i0, j0, k0, tx, ty, tz))
        return false;

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    // Need i0-1..i0+2 in each direction.
    if (i0 < I0 + 1 || i0 + 2 >= I1 || j0 < J0 + 1 || j0 + 2 >= J1 || k0 < K0 + 1 ||
        k0 + 2 >= K1) {
        return false;
    }

    double wx[4], wy[4], wz[4];
    lagrange4_weights(tx, wx);
    lagrange4_weights(ty, wy);
    lagrange4_weights(tz, wz);

    value = 0.0;
    for (size_t a = 0; a < 4; ++a) {
        const size_t ii = i0 + a - 1;
        for (size_t b = 0; b < 4; ++b) {
            const size_t jj = j0 + b - 1;
            for (size_t c = 0; c < 4; ++c) {
                const size_t kk = k0 + c - 1;
                const double v = double(field.ptr()[field.idx(ii, jj, kk)]);
                if (!std::isfinite(v))
                    return false;
                value += wx[a] * wy[b] * wz[c] * v;
            }
        }
    }
    return std::isfinite(value);
}

template <typename FieldLike>
bool sample_scalar_tracker_interp(const tensorium::tests::Grid &grid, const FieldLike &field,
                                  double x, double y, double z, double &value) {
    if (sample_scalar_lagrange4(grid, field, x, y, z, value))
        return true;
    return sample_scalar_trilinear(grid, field, x, y, z, value);
}

bool sample_vector3_trilinear(const tensorium::tests::Grid &grid, const tensorium_RG::Field3D<double> f[3],
                              const std::array<double, 3> &p, std::array<double, 3> &out) {
    for (int d = 0; d < 3; ++d) {
        if (!sample_scalar_trilinear(grid, f[d], p[0], p[1], p[2], out[d]))
            return false;
    }
    return true;
}

bool sample_vector3_tracker_interp(const tensorium::tests::Grid                &grid,
                                   const tensorium_RG::Field3D<double>          f[3],
                                   const std::array<double, 3>                 &p,
                                   std::array<double, 3>                       &out) {
    for (int d = 0; d < 3; ++d) {
        if (!sample_scalar_tracker_interp(grid, f[d], p[0], p[1], p[2], out[d]))
            return false;
    }
    return true;
}

void clamp_tracker_to_domain(const tensorium::tests::Grid &grid, std::array<double, 3> &p) {
    const double x_min = grid.x0;
    const double y_min = grid.y0;
    const double z_min = grid.z0;
    const double x_max = grid.x0 + (double(grid.dims.nx) - 1.0) * grid.dx;
    const double y_max = grid.y0 + (double(grid.dims.ny) - 1.0) * grid.dy;
    const double z_max = grid.z0 + (double(grid.dims.nz) - 1.0) * grid.dz;
    p[0] = std::min(std::max(p[0], x_min), x_max);
    p[1] = std::min(std::max(p[1], y_min), y_max);
    p[2] = std::min(std::max(p[2], z_min), z_max);
}

void initialize_shift_puncture_tracker(const tensorium::tests::Grid &grid, ShiftPunctureTracker &tracker,
                                       const std::array<double, 3> &p1,
                                       const std::array<double, 3> &p2) {
    tracker = ShiftPunctureTracker{};
    tracker.p1 = p1;
    tracker.p2 = p2;
    clamp_tracker_to_domain(grid, tracker.p1);
    clamp_tracker_to_domain(grid, tracker.p2);

    std::array<double, 3> b1{0.0, 0.0, 0.0};
    std::array<double, 3> b2{0.0, 0.0, 0.0};
    (void)sample_vector3_tracker_interp(grid, grid.beta, tracker.p1, b1);
    (void)sample_vector3_tracker_interp(grid, grid.beta, tracker.p2, b2);
    tracker.beta1_prev = b1;
    tracker.beta2_prev = b2;
    tracker.initialized = true;
}

void advance_shift_puncture_tracker(const tensorium::tests::Grid &grid, ShiftPunctureTracker &tracker,
                                    double dt) {
    if (!tracker.initialized || !std::isfinite(dt) || dt <= 0.0)
        return;

    std::array<double, 3> b1_new{0.0, 0.0, 0.0};
    std::array<double, 3> b2_new{0.0, 0.0, 0.0};
    if (!sample_vector3_tracker_interp(grid, grid.beta, tracker.p1, b1_new))
        b1_new = tracker.beta1_prev;
    if (!sample_vector3_tracker_interp(grid, grid.beta, tracker.p2, b2_new))
        b2_new = tracker.beta2_prev;

    for (int d = 0; d < 3; ++d) {
        tracker.p1[d] += -0.5 * dt * (tracker.beta1_prev[d] + b1_new[d]);
        tracker.p2[d] += -0.5 * dt * (tracker.beta2_prev[d] + b2_new[d]);
    }
    clamp_tracker_to_domain(grid, tracker.p1);
    clamp_tracker_to_domain(grid, tracker.p2);
    tracker.beta1_prev = b1_new;
    tracker.beta2_prev = b2_new;
}

PuncturePlaneSample make_tracker_sample(const tensorium::tests::Grid &grid,
                                        const ShiftPunctureTracker &tracker) {
    PuncturePlaneSample sample;
    if (!tracker.initialized)
        return sample;

    double chi1 = std::numeric_limits<double>::quiet_NaN();
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    double alpha1 = std::numeric_limits<double>::quiet_NaN();
    double alpha2 = std::numeric_limits<double>::quiet_NaN();
    (void)sample_scalar_tracker_interp(grid, grid.chi, tracker.p1[0], tracker.p1[1], tracker.p1[2],
                                       chi1);
    (void)sample_scalar_tracker_interp(grid, grid.chi, tracker.p2[0], tracker.p2[1], tracker.p2[2],
                                       chi2);
    (void)sample_scalar_tracker_interp(grid, grid.alpha, tracker.p1[0], tracker.p1[1],
                                       tracker.p1[2], alpha1);
    (void)sample_scalar_tracker_interp(grid, grid.alpha, tracker.p2[0], tracker.p2[1],
                                       tracker.p2[2], alpha2);

    sample.has_left = true;
    sample.x_left = tracker.p1[0];
    sample.y_left = tracker.p1[1];
    sample.chi_left = chi1;
    sample.alpha_left = alpha1;

    sample.has_right = true;
    sample.x_right = tracker.p2[0];
    sample.y_right = tracker.p2[1];
    sample.chi_right = chi2;
    sample.alpha_right = alpha2;
    return sample;
}

PuncturePlaneSample sample_punctures_equatorial(const tensorium::tests::Grid &grid,
                                                const PunctureTrackHistory *history = nullptr,
                                                double current_time = 0.0,
                                                TrackerRecenterMode score_mode =
                                                    TrackerRecenterMode::Mixed) {
    PuncturePlaneSample sample;
    const size_t ng = grid.dims.ng;
    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t i_min = ng;
    const size_t i_max = ng + nx - 1;
    const size_t j_min = ng;
    const size_t j_max = ng + ny - 1;
    const double h = std::max(grid.dx, grid.dy);
    const bool have_prev = (history != nullptr && history->has_last && history->last.has_left &&
                            history->last.has_right && std::isfinite(history->last.x_left) &&
                            std::isfinite(history->last.y_left) &&
                            std::isfinite(history->last.x_right) &&
                            std::isfinite(history->last.y_right));
    const PuncturePrediction pred =
        (history != nullptr) ? predict_puncture_positions(*history, current_time)
                             : PuncturePrediction{};

    constexpr size_t kLocalRadiusCells = 14;
    constexpr size_t kDistinctRadiusCells = 5;
    constexpr double kDistWeight = 3e-3;
    constexpr double kAlphaWeight = 5e-2;

    auto local_search = [&](double x_ref, double y_ref, bool with_exclusion = false,
                            double ex_x = 0.0, double ex_y = 0.0,
                            double ex_r2 = 0.0) -> PunctureCandidate {
        const size_t ic = nearest_index(x_ref, grid.x0, grid.dx, i_min, i_max);
        const size_t jc = nearest_index(y_ref, grid.y0, grid.dy, j_min, j_max);

        const size_t i_lo = (ic > kLocalRadiusCells) ? ic - kLocalRadiusCells : i_min;
        const size_t j_lo = (jc > kLocalRadiusCells) ? jc - kLocalRadiusCells : j_min;
        const size_t i_hi = std::min(i_max, ic + kLocalRadiusCells) + 1;
        const size_t j_hi = std::min(j_max, jc + kLocalRadiusCells) + 1;
        return find_min_candidate_in_window(grid, i_lo, i_hi, j_lo, j_hi, with_exclusion, ex_x,
                                            ex_y, ex_r2, true, x_ref, y_ref, kDistWeight,
                                            kAlphaWeight, score_mode);
    };

    PunctureCandidate a;
    PunctureCandidate b;
    bool have_pair = false;

    if (pred.valid) {
        a = local_search(pred.x_left, pred.y_left);
        const double ex_r = double(kDistinctRadiusCells) * h;
        b = local_search(pred.x_right, pred.y_right, true, a.x, a.y, ex_r * ex_r);
        have_pair = (a.found && b.found);
    }

    if (!have_pair && have_prev) {
        a = local_search(history->last.x_left, history->last.y_left);
        const double ex_r = double(kDistinctRadiusCells) * h;
        b = local_search(history->last.x_right, history->last.y_right, true, a.x, a.y,
                         ex_r * ex_r);
        have_pair = (a.found && b.found);
    }

    if (!have_pair) {
        a = find_min_candidate_in_window(grid, i_min, i_max + 1, j_min, j_max + 1, false, 0.0,
                                         0.0, 0.0, false, 0.0, 0.0, kDistWeight, kAlphaWeight,
                                         score_mode);
        if (a.found) {
            const double ex_r = double(kDistinctRadiusCells) * h;
            b = find_min_candidate_in_window(grid, i_min, i_max + 1, j_min, j_max + 1, true, a.x,
                                             a.y, ex_r * ex_r, false, 0.0, 0.0, kDistWeight,
                                             kAlphaWeight, score_mode);
            have_pair = b.found;
        }
    }

    if (!have_pair)
        return sample;

    refine_candidate_xy(grid, a);
    refine_candidate_xy(grid, b);

    PunctureCandidate left = a;
    PunctureCandidate right = b;
    if (have_prev) {
        const auto dist = [](double xr, double yr, const PunctureCandidate &c) {
            const double dx = c.x - xr;
            const double dy = c.y - yr;
            return std::sqrt(dx * dx + dy * dy);
        };
        const double xL_ref = pred.valid ? pred.x_left : history->last.x_left;
        const double yL_ref = pred.valid ? pred.y_left : history->last.y_left;
        const double xR_ref = pred.valid ? pred.x_right : history->last.x_right;
        const double yR_ref = pred.valid ? pred.y_right : history->last.y_right;
        const double keep = dist(xL_ref, yL_ref, a) + dist(xR_ref, yR_ref, b);
        const double swap = dist(xL_ref, yL_ref, b) + dist(xR_ref, yR_ref, a);
        if (swap < keep) {
            left = b;
            right = a;
        }

        // Keep continuity without freezing: clamp over-large jumps toward previous positions.
        const double max_jump = 6.0 * h;
        (void)clamp_jump_to_radius(history->last.x_left, history->last.y_left, max_jump, left.x,
                                   left.y);
        (void)clamp_jump_to_radius(history->last.x_right, history->last.y_right, max_jump,
                                   right.x, right.y);
    } else if (a.x > b.x) {
        left = b;
        right = a;
    }

    sample.has_left = true;
    sample.x_left = left.x;
    sample.y_left = left.y;
    sample.chi_left = left.chi;
    sample.alpha_left = left.alpha;

    sample.has_right = true;
    sample.x_right = right.x;
    sample.y_right = right.y;
    sample.chi_right = right.chi;
    sample.alpha_right = right.alpha;
    return sample;
}

using tensorium::tests::Grid;

void initialize_single_boost(Grid &grid, double separation, double tangential_momentum,
                             double radial_momentum, double m1, double m2) {
    // Keep punctures on the y-axis and use momentum components in the x-y plane.
    const double x = 0.0;
    const double y1 = separation;
    const double y2 = -separation;
    const double z = 0.0;

    // The "tangential" value maps to Px and the "radial" value maps to Py.
    // Signs are chosen so the binary starts with opposite linear momenta.
    const double P1[3] = {-tangential_momentum, -radial_momentum, 0.0};
    const double P2[3] = {tangential_momentum, radial_momentum, 0.0};
    const double S1[3] = {0.0, 0.0, 0.0};
    const double S2[3] = {0.0, 0.0, 0.0};

    tensorium_RG::init::binary_bowen_york_puncture_init(grid, m1, x, y1, z, P1, S1, m2, x, y2, z,
                                                        P2, S2, 1e-10);
}

void initialize_single_boost_interpolated(Grid &grid, double separation,
                                          double tangential_momentum, double radial_momentum,
                                          double m1, double m2, size_t interp_seed_n) {
    // TwoPunctures-style interpolated data assume punctures centered on the x-axis.
    const double x1 = -separation;
    const double x2 = separation;
    const double y = 0.0;
    const double z = 0.0;

    // Match TwoPunctures conventions used in the GRChombo BinaryBH TP setup:
    // P_minus = (+Pr, -Pt, 0), P_plus = (-Pr, +Pt, 0).
    const double P1[3] = {radial_momentum, -tangential_momentum, 0.0}; // x1 = -separation
    const double P2[3] = {-radial_momentum, tangential_momentum, 0.0}; // x2 = +separation
    const double S1[3] = {0.0, 0.0, 0.0};
    const double S2[3] = {0.0, 0.0, 0.0};

    tensorium_RG::init::binary_bowen_york_puncture_interpolated_init(
        grid, m1, x1, y, z, P1, S1, m2, x2, y, z, P2, S2, interp_seed_n, 1e-10);
}

struct ScopedEnvOverride {
    explicit ScopedEnvOverride(const char *key_, const char *value) : key(key_) {
        const char *old = std::getenv(key);
        if (old != nullptr) {
            had_old = true;
            old_value = old;
        }
        setenv(key, value, 1);
    }

    ~ScopedEnvOverride() {
        if (had_old)
            setenv(key, old_value.c_str(), 1);
        else
            unsetenv(key);
    }

    const char *key;
    bool        had_old = false;
    std::string old_value;
};

} // namespace

REGISTER_TEST(
    "bssn.viz.moving_puncture", "Export CSV slices of a moving spinning black hole", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 96;
        cfg.ny = 96;
        cfg.nz = 96;
        cfg.spacing = 62.4 / 96.0;
        cfg.ng = 6;
        cfg.padding = 0;
        cfg.steps = 6000;
        cfg.cfl = 0.10;
        cfg.gauge_factor = 1.0;

        // Physical box length used by default when spacing is not overridden explicitly.
        // With defaults: L = 62.4
        double box_length = cfg.spacing * static_cast<double>(cfg.nx);

        if (const char *n_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_GRID_N")) {
            const long parsed = std::strtol(n_env, nullptr, 10);
            if (parsed > 8) {
                const size_t n = static_cast<size_t>(parsed);
                cfg.nx = n;
                cfg.ny = n;
                cfg.nz = n;
            }
        }
        if (const char *nx_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_NX")) {
            const long parsed = std::strtol(nx_env, nullptr, 10);
            if (parsed > 8)
                cfg.nx = static_cast<size_t>(parsed);
        }
        if (const char *ny_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_NY")) {
            const long parsed = std::strtol(ny_env, nullptr, 10);
            if (parsed > 8)
                cfg.ny = static_cast<size_t>(parsed);
        }
        if (const char *nz_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_NZ")) {
            const long parsed = std::strtol(nz_env, nullptr, 10);
            if (parsed > 8)
                cfg.nz = static_cast<size_t>(parsed);
        }
        if (const char *box_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH")) {
            char *end = nullptr;
            const double parsed = std::strtod(box_env, &end);
            if (end != box_env && std::isfinite(parsed) && parsed > 0.0)
                box_length = parsed;
        }
        bool spacing_set = false;
        if (const char *spacing_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_SPACING")) {
            char *end = nullptr;
            const double parsed = std::strtod(spacing_env, &end);
            if (end != spacing_env && std::isfinite(parsed) && parsed > 0.0) {
                cfg.spacing = parsed;
                spacing_set = true;
            }
        }
        if (!spacing_set)
            cfg.spacing = box_length / static_cast<double>(cfg.nx);

        if (const char *steps_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_STEPS")) {
            const long parsed = std::strtol(steps_env, nullptr, 10);
            if (parsed > 0)
                cfg.steps = static_cast<size_t>(parsed);
        }
        if (const char *cfl_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_CFL")) {
            char *end = nullptr;
            const double parsed = std::strtod(cfl_env, &end);
            if (end != cfl_env && std::isfinite(parsed) && parsed > 0.0)
                cfg.cfl = parsed;
        }

        std::cout << "[mesh] nx=" << cfg.nx << " ny=" << cfg.ny << " nz=" << cfg.nz
                  << " spacing=" << cfg.spacing
                  << " box=(" << cfg.nx * cfg.spacing << ", " << cfg.ny * cfg.spacing << ", "
                  << cfg.nz * cfg.spacing << ")" << std::endl;

        int spatial_derivative_order = 4;
        if (const char *order_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_SPATIAL_ORDER")) {
            const int parsed = std::atoi(order_env);
            if (parsed == 4 || parsed == 6)
                spatial_derivative_order = parsed;
        }
        tensorium_RG::fd::set_max_spatial_derivative_order(spatial_derivative_order);
        std::cout << "[num] spatial_derivative_order=" << spatial_derivative_order << std::endl;

        tensorium_RG::fd::set_fd_dx(cfg.spacing);
        (void)system("mkdir -p Output/viz");

        tensorium::tests::Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing,
                                    cfg.spacing);

        grid.x0 = -0.5 * cfg.spacing * cfg.nx + 0.5 * cfg.spacing;
        grid.y0 = -0.5 * cfg.spacing * cfg.ny + 0.5 * cfg.spacing;
        grid.z0 = -0.5 * cfg.spacing * cfg.nz + 0.5 * cfg.spacing;

        double m1 = 0.48847892320123;
        double m2 = 0.48847892320123;
        double separation = 6.10679;
        double momentum = 0.0841746;
        double radial_momentum = 0.000510846;
        if (const char *mass_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_MASS")) {
            char *end = nullptr;
            const double parsed = std::strtod(mass_env, &end);
            if (end != mass_env && std::isfinite(parsed) && parsed > 0.0) {
                m1 = parsed;
                m2 = parsed;
            }
        }
        if (const char *mass1_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_MASS1")) {
            char *end = nullptr;
            const double parsed = std::strtod(mass1_env, &end);
            if (end != mass1_env && std::isfinite(parsed) && parsed > 0.0)
                m1 = parsed;
        }
        if (const char *mass2_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_MASS2")) {
            char *end = nullptr;
            const double parsed = std::strtod(mass2_env, &end);
            if (end != mass2_env && std::isfinite(parsed) && parsed > 0.0)
                m2 = parsed;
        }
        if (const char *sep_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_SEPARATION")) {
            char *end = nullptr;
            const double parsed = std::strtod(sep_env, &end);
            if (end != sep_env && std::isfinite(parsed) && parsed > 0.0)
                separation = parsed;
        }
        if (const char *mom_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_MOMENTUM")) {
            char *end = nullptr;
            const double parsed = std::strtod(mom_env, &end);
            if (end != mom_env && std::isfinite(parsed) && parsed > 0.0)
                momentum = parsed;
        }
        if (const char *mom_r_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_RADIAL_MOMENTUM")) {
            char *end = nullptr;
            const double parsed = std::strtod(mom_r_env, &end);
            if (end != mom_r_env && std::isfinite(parsed) && parsed >= 0.0)
                radial_momentum = parsed;
        }
        const double user_tangential_momentum = momentum;
        const CircularMomentumSuggestion circular_hint =
            suggest_circular_momentum(m1, m2, separation);
        const bool print_suggested_momentum =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_PRINT_SUGGESTED_MOMENTUM", false);
        const bool auto_circular =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_AUTO_CIRCULAR", false);

        if (print_suggested_momentum) {
            if (circular_hint.valid) {
                std::cout << "[Z4c.Init] Physics Diagnostic for d = " << circular_hint.d
                          << std::endl;
                std::cout << "[Z4c.Init] > Suggested P_tang (Newtonian): "
                          << circular_hint.p_newtonian << std::endl;
                std::cout << "[Z4c.Init] > Suggested P_tang (Post-Newtonian): "
                          << circular_hint.p_pn << std::endl;
                std::cout << "[Z4c.Init] Current P_tang (User): " << user_tangential_momentum
                          << std::endl;
                std::cout
                    << "[Z4c.Init] Note: suggestions are optimized for Z4c with active "
                       "constraint damping."
                    << std::endl;
            } else {
                std::cout << "[Z4c.Init] Physics Diagnostic unavailable: invalid masses/"
                             "separation for momentum suggestion."
                          << std::endl;
            }
        }

        if (auto_circular) {
            if (circular_hint.valid) {
                momentum = circular_hint.p_pn;
                std::cout << "[Z4c.Init] AUTO_CIRCULAR enabled: overriding P_tang from "
                          << user_tangential_momentum << " to " << momentum
                          << " (Post-Newtonian, Z4c constraint-damping preset)." << std::endl;
            } else {
                std::cout << "[Z4c.Init] AUTO_CIRCULAR requested but suggestion is invalid; "
                             "keeping user P_tang="
                          << momentum << std::endl;
            }
        }
        std::cout << "[init] puncture separation=" << separation
                  << " momentum_tan=" << momentum
                  << " momentum_rad=" << radial_momentum
                  << " m1=" << m1
                  << " m2=" << m2 << std::endl;
        const bool use_interpolated_init =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT", false);
        size_t interp_seed_n = 64;
        if (const char *seed_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_INTERP_SEED_N")) {
            const long parsed = std::strtol(seed_env, nullptr, 10);
            if (parsed >= 24)
                interp_seed_n = static_cast<size_t>(parsed);
        }
        const char *interp_mode = "interpolated_seed";
#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
        interp_mode = "interpolated_twopunctures";
#endif
        std::cout << "[init] mode=" << (use_interpolated_init ? interp_mode : "bowen_york")
                  << " seed_n=" << interp_seed_n << std::endl;
        if (use_interpolated_init)
            initialize_single_boost_interpolated(grid, separation, momentum, radial_momentum, m1,
                                                 m2, interp_seed_n);
        else
            initialize_single_boost(grid, separation, momentum, radial_momentum, m1, m2);

        tensorium_RG::bssn::ProjectionConfig proj_cfg;
        proj_cfg.padding = 0;
        proj_cfg.renormalize_metric = true;
        proj_cfg.project_A_tilde = true;
        proj_cfg.recompute_inverse = true;

        tensorium_RG::bssn::GaugeParameters<double> params;
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

        // Point 5 parity: keep moving-puncture gauge strictly aligned with
        // two-puncture reference in interpolated mode unless explicitly disabled.
        const bool strict_tp_gauge =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_STRICT_TP_GAUGE",
                              use_interpolated_init);
        if (const char *shift_eta_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_SHIFT_ETA")) {
            char *end = nullptr;
            const double parsed = std::strtod(shift_eta_env, &end);
            if (end != shift_eta_env && std::isfinite(parsed) && parsed >= 0.0) {
                params.eta = parsed;
                params.shift_eta = parsed;
            }
        }
        if (const char *kappa1_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_KAPPA1")) {
            char *end = nullptr;
            const double parsed = std::strtod(kappa1_env, &end);
            if (end != kappa1_env && std::isfinite(parsed) && parsed >= 0.0)
                params.kappa1 = parsed;
        }
        if (const char *kappa2_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_KAPPA2")) {
            char *end = nullptr;
            const double parsed = std::strtod(kappa2_env, &end);
            if (end != kappa2_env && std::isfinite(parsed))
                params.kappa2 = parsed;
        }
        if (const char *kappa3_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_KAPPA3")) {
            char *end = nullptr;
            const double parsed = std::strtod(kappa3_env, &end);
            if (end != kappa3_env && std::isfinite(parsed))
                params.kappa3 = parsed;
        }
        params.covariant_z4 =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_COVARIANT_Z4", params.covariant_z4);
        params.evolve_Z =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EVOLVE_Z", params.evolve_Z);
        if (const char *chi_div_floor_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_CHI_DIV_FLOOR")) {
            char *end = nullptr;
            const double parsed = std::strtod(chi_div_floor_env, &end);
            if (end != chi_div_floor_env && std::isfinite(parsed) && parsed > 0.0)
                params.chi_div_floor = parsed;
        }
        if (const char *ko_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_KO_SIGMA")) {
            char *end = nullptr;
            const double parsed = std::strtod(ko_env, &end);
            if (end != ko_env && std::isfinite(parsed) && parsed >= 0.0)
                params.ko_sigma = parsed;
        }
        if (const char *alpha_floor_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_ALPHA_FLOOR")) {
            char *end = nullptr;
            const double parsed = std::strtod(alpha_floor_env, &end);
            if (end != alpha_floor_env && std::isfinite(parsed) && parsed >= 0.0)
                params.alpha_floor = parsed;
        }
        if (const char *chi_floor_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_CHI_FLOOR")) {
            char *end = nullptr;
            const double parsed = std::strtod(chi_floor_env, &end);
            if (end != chi_floor_env && std::isfinite(parsed) && parsed >= 0.0)
                params.chi_floor = parsed;
        }
        size_t projection_stride = 0;
        if (const char *proj_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_PROJECTION_STRIDE")) {
            const long parsed = std::strtol(proj_env, nullptr, 10);
            if (parsed > 0)
                projection_stride = static_cast<size_t>(parsed);
        }
        if (const char *ultra_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_ULTRA_STABLE")) {
            if (std::atoi(ultra_env) != 0) {
                cfg.cfl = std::min(cfg.cfl, 0.07);
                params.eta = std::max(params.eta, 6.0);
                params.shift_eta = std::max(params.shift_eta, 6.0);
                params.kappa1 = std::max(params.kappa1, 0.18);
                params.ko_sigma = std::max(params.ko_sigma, 0.28);
                projection_stride =
                    (projection_stride == 0) ? size_t(5) : std::min(projection_stride, size_t(5));
            }
        }

        if (strict_tp_gauge) {
            params.use_direct_shift_rhs = false;
            // Match GRChombo BinaryBH two-puncture gauge defaults.
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
            std::cout << "[gauge] strict_tp_gauge=1 (two-puncture reference preset)"
                      << std::endl;
        } else {
            std::cout << "[gauge] strict_tp_gauge=0 (env-tuned gauge allowed)" << std::endl;
        }
        std::cout << "[scale] alpha_floor=" << params.alpha_floor
                  << " chi_floor=" << params.chi_floor
                  << " chi_div_floor=" << params.chi_div_floor
                  << " min_lapse_for_K=" << params.min_lapse_for_K << std::endl;

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
        rhs_ix1 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX1", rhs_ix1);
        rhs_ox1 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX1", rhs_ox1);
        rhs_ix2 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX2", rhs_ix2);
        rhs_ox2 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX2", rhs_ox2);
        rhs_ix3 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX3", rhs_ix3);
        rhs_ox3 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX3", rhs_ox3);
        rf_ix1 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX1", rf_ix1);
        rf_ox1 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX1", rf_ox1);
        rf_ix2 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX2", rf_ix2);
        rf_ox2 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX2", rf_ox2);
        rf_ix3 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX3", rf_ix3);
        rf_ox3 = parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX3", rf_ox3);
        tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(rf_ix1, rf_ox1, rf_ix2, rf_ox2,
                                                                    rf_ix3, rf_ox3);
        tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(
            rhs_ix1, rhs_ox1, rhs_ix2, rhs_ox2, rhs_ix3, rhs_ox3);
        std::cout << "[bc] reflective_faces="
                  << " ix1=" << rf_ix1 << " ox1=" << rf_ox1
                  << " ix2=" << rf_ix2 << " ox2=" << rf_ox2
                  << " ix3=" << rf_ix3 << " ox3=" << rf_ox3 << std::endl;
        std::cout << "[bc] rhs_sommerfeld_faces="
                  << " ix1=" << rhs_ix1 << " ox1=" << rhs_ox1
                  << " ix2=" << rhs_ix2 << " ox2=" << rhs_ox2
                  << " ix3=" << rhs_ix3 << " ox3=" << rhs_ox3 << std::endl;

        tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);
        tensorium_RG::init::zero_z4c_fields(grid);

        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, cfg.padding);
        stepper.set_gauge_parameters(params);
        size_t state_log_stride = 10;
        if (const char *state_log_stride_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_STATE_LOG_STRIDE")) {
            const long parsed = std::strtol(state_log_stride_env, nullptr, 10);
            if (parsed > 0)
                state_log_stride = static_cast<size_t>(parsed);
        }
        stepper.set_state_log_stride(state_log_stride);
        std::cout << "[log] state_log_stride=" << state_log_stride << std::endl;

        ShiftPunctureTracker puncture_shift_tracker;
        if (use_interpolated_init) {
            initialize_shift_puncture_tracker(
                grid, puncture_shift_tracker,
                std::array<double, 3>{-separation, 0.0, 0.0},
                std::array<double, 3>{separation, 0.0, 0.0});
        } else {
            initialize_shift_puncture_tracker(
                grid, puncture_shift_tracker,
                std::array<double, 3>{0.0, separation, 0.0},
                std::array<double, 3>{0.0, -separation, 0.0});
        }
        std::cout << "[tracker] mode=shift_integrate order=2" << std::endl;

        const size_t export_stride = 10;
        stepper.set_snapshot_callback(
            [export_stride](const tensorium::tests::Grid &g, size_t step) {
                if (step % export_stride == 0) {
                    std::cout << ">> Exporting slice " << step << "..." << std::endl;
                    export_slice_csv(g, step, "Output/viz");
                }
            });

        auto constraint_scratch = tensorium::tests::make_constraint_scratch(grid);
        double theta_cap = 8e-2;
        double constraint_cap = 6e-1;
        bool strict_guards = false;
        if (const char *strict_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_STRICT_GUARDS")) {
            strict_guards = (std::atoi(strict_env) != 0);
        }
        if (const char *theta_cap_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_THETA_CAP")) {
            char *end = nullptr;
            const double parsed = std::strtod(theta_cap_env, &end);
            if (end != theta_cap_env && std::isfinite(parsed) && parsed > 0.0)
                theta_cap = parsed;
        }
        if (const char *constraint_cap_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_CAP")) {
            char *end = nullptr;
            const double parsed = std::strtod(constraint_cap_env, &end);
            if (end != constraint_cap_env && std::isfinite(parsed) && parsed > 0.0)
                constraint_cap = parsed;
        }
        // Optional relaxed mode for exploratory long runs.
        if (const char *relaxed_env = std::getenv("TENSORIUM_MOVING_PUNCTURE_RELAXED_GUARDS")) {
            if (std::atoi(relaxed_env) != 0) {
                theta_cap = std::max(theta_cap, 1.2e-1);
                constraint_cap = std::max(constraint_cap, 8e-1);
            }
        }
        auto collect_diagnostics =
            [&]() -> tensorium_RG::bssn::ConstraintMonitorStats {
            const double r_min = 2.0 * cfg.spacing;
            const double r_max = 0.45 * cfg.spacing * (cfg.nx - 1);
            tensorium_RG::bssn::compute_bssn_constraints(grid, grid.Ricci, constraint_scratch.H,
                                                         constraint_scratch.M, constraint_scratch.C,
                                                         r_min, r_max, 0.0, 0.0, 0.0, false);
            auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, constraint_scratch.H,
                                                                        cfg.padding);
            tensorium_RG::bssn::populate_constraint_norms(grid, constraint_scratch.M, stats,
                                                          cfg.padding);
            return stats;
        };
        auto compute_constraints_full_domain_for_slice = [&]() {
            tensorium_RG::bssn::compute_bssn_constraints(
                grid, grid.Ricci, constraint_scratch.H, constraint_scratch.M, constraint_scratch.C,
                0.0, std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);
        };
        auto log_diagnostics = [&](size_t step_index,
                                   const tensorium_RG::bssn::ConstraintMonitorStats *precomputed =
                                       nullptr) {
            const auto stats = precomputed ? *precomputed : collect_diagnostics();
            std::cout << "[diag] step=" << step_index << " ||Theta||2=" << stats.l2_theta
                      << " ||Z||2=" << stats.l2_Z << " ||H||2=" << stats.l2_H
                      << " ||M||2=" << stats.l2_M << " det_drift=" << stats.max_det_drift
                      << " trA=" << stats.max_trace_A << " caps(theta=" << theta_cap
                      << ",constraint=" << constraint_cap << ")" << std::endl;
            const bool exceeded = (stats.l2_theta > theta_cap || stats.l2_Z > theta_cap ||
                                   stats.l2_H > constraint_cap || stats.l2_M > constraint_cap);
            if (exceeded) {
                if (strict_guards) {
                    std::cout << "[warn] aborting run due to constraint growth" << std::endl;
                    return false;
                }
                std::cout << "[warn] constraint growth above cap, continuing (strict guards off)"
                          << std::endl;
            }
            return true;
        };

        auto guard_gauge = [&](size_t step_index) {
            size_t i0, i1, j0, j1, k0, k1;
            tensorium::tests::interior_bounds(grid, cfg.padding, i0, i1, j0, j1, k0, k1);
            double alpha_min = std::numeric_limits<double>::infinity();
            double alpha_min_nonpuncture = std::numeric_limits<double>::infinity();
            double chi_min = std::numeric_limits<double>::infinity();
            constexpr double chi_puncture_cut = 5e-3;
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    for (size_t k = k0; k < k1; ++k) {
                        const size_t idx = grid.alpha.idx(i, j, k);
                        const double alpha = double(grid.alpha.ptr()[idx]);
                        const double chi = double(grid.chi.ptr()[idx]);
                        if (!std::isfinite(alpha) || !std::isfinite(chi)) {
                            std::cout << "[warn] non-finite gauge values at step=" << step_index
                                      << std::endl;
                            if (strict_guards)
                                return false;
                            continue;
                        }
                        alpha_min = std::min(alpha_min, alpha);
                        chi_min = std::min(chi_min, chi);
                        if (chi > chi_puncture_cut)
                            alpha_min_nonpuncture = std::min(alpha_min_nonpuncture, alpha);
                    }
            if (chi_min <= 0.0 ||
                alpha_min_nonpuncture < 1e-4) {
                std::cout << "[warn] gauge collapse alpha_min=" << alpha_min
                          << " alpha_min_nonpuncture=" << alpha_min_nonpuncture
                          << " chi_min=" << chi_min << " at step=" << step_index << std::endl;
                if (strict_guards)
                    return false;
            }
            return true;
        };

        double t = 0.0;

        const size_t log_stride = 20;
        size_t       constraint_export_stride = 5;
        if (const char *cstride_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_EXPORT_STRIDE")) {
            const long parsed = std::strtol(cstride_env, nullptr, 10);
            if (parsed > 0)
                constraint_export_stride = static_cast<size_t>(parsed);
        }
        bool export_constraint_slices =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EXPORT_CONSTRAINT_SLICES", true);
        size_t constraint_slice_stride = constraint_export_stride;
        if (const char *slice_stride_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_SLICE_STRIDE")) {
            const long parsed = std::strtol(slice_stride_env, nullptr, 10);
            if (parsed > 0)
                constraint_slice_stride = static_cast<size_t>(parsed);
        }
        (void)std::remove("Output/viz/puncture_track.csv");
        std::ofstream puncture_track("Output/viz/puncture_track.csv",
                                     std::ios::out | std::ios::trunc);
        if (puncture_track.is_open()) {
            puncture_track.setf(std::ios::unitbuf);
            puncture_track << "step,t,"
                           << "x_left,y_left,chi_left,alpha_left,"
                           << "x_right,y_right,chi_right,alpha_right\n";
        } else {
            std::cout << "[warn] could not open Output/viz/puncture_track.csv for writing"
                      << std::endl;
        }
        (void)std::remove("Output/viz/constraints_norms.csv");
        std::ofstream constraint_log("Output/viz/constraints_norms.csv",
                                     std::ios::out | std::ios::trunc);
        if (constraint_log.is_open()) {
            constraint_log.setf(std::ios::unitbuf);
            constraint_log
                << "step,t,dt,l2_theta,l2_Z,l2_H,l2_M,max_H,max_det_drift,max_trace_A,samples\n";
        } else {
            std::cout << "[warn] could not open Output/viz/constraints_norms.csv for writing"
                      << std::endl;
        }
        std::cout << "[constraints] norms_stride=" << constraint_export_stride
                  << " slice_export=" << (export_constraint_slices ? 1 : 0)
                  << " slice_stride=" << constraint_slice_stride << std::endl;

        (void)std::remove("Output/viz/puncture_track_minima.csv");
        std::ofstream puncture_track_minima("Output/viz/puncture_track_minima.csv",
                                            std::ios::out | std::ios::trunc);
        if (puncture_track_minima.is_open()) {
            puncture_track_minima.setf(std::ios::unitbuf);
            puncture_track_minima << "step,t,"
                                  << "x_left,y_left,chi_left,alpha_left,"
                                  << "x_right,y_right,chi_right,alpha_right\n";
        } else {
            std::cout << "[warn] could not open Output/viz/puncture_track_minima.csv for writing"
                      << std::endl;
        }

        (void)std::remove("Output/viz/puncture_tracker_drift.csv");
        std::ofstream puncture_tracker_drift("Output/viz/puncture_tracker_drift.csv",
                                             std::ios::out | std::ios::trunc);
        if (puncture_tracker_drift.is_open()) {
            puncture_tracker_drift.setf(std::ios::unitbuf);
            puncture_tracker_drift
                << "step,t,drift_left,drift_right,recentered,"
                << "shift_x_left,shift_y_left,shift_x_right,shift_y_right,"
                << "min_x_left,min_y_left,min_x_right,min_y_right\n";
        } else {
            std::cout << "[warn] could not open Output/viz/puncture_tracker_drift.csv for writing"
                      << std::endl;
        }

        PunctureTrackHistory puncture_minima_history;
        const TrackerRecenterMode tracker_recenter_mode = parse_tracker_recenter_mode_or(
            "TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_MODE", TrackerRecenterMode::Mixed);
        {
            const auto init_minima =
                sample_punctures_equatorial(grid, nullptr, t, tracker_recenter_mode);
            if (init_minima.has_left && init_minima.has_right) {
                puncture_minima_history.last = init_minima;
                puncture_minima_history.t_last = t;
                puncture_minima_history.has_last = true;
            }
        }

        const double tracker_h = std::max(grid.dx, grid.dy);
        double       tracker_drift_warn_cells = 1.5;
        if (const char *warn_cells_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_TRACKER_DRIFT_WARN_CELLS")) {
            char *end = nullptr;
            const double parsed = std::strtod(warn_cells_env, &end);
            if (end != warn_cells_env && std::isfinite(parsed) && parsed > 0.0)
                tracker_drift_warn_cells = parsed;
        }
        bool tracker_recenter_on_drift =
            parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_ON_DRIFT", false);
        double tracker_recenter_cells = 4.0;
        if (const char *recenter_cells_env =
                std::getenv("TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_CELLS")) {
            char *end = nullptr;
            const double parsed = std::strtod(recenter_cells_env, &end);
            if (end != recenter_cells_env && std::isfinite(parsed) && parsed > 0.0)
                tracker_recenter_cells = parsed;
        }
        const double tracker_drift_warn_radius = tracker_drift_warn_cells * tracker_h;
        const double tracker_recenter_radius = tracker_recenter_cells * tracker_h;
        std::cout << "[tracker] interp=lagrange4(fallback=trilinear) drift_warn_cells="
                  << tracker_drift_warn_cells << " recenter_on_drift="
                  << (tracker_recenter_on_drift ? 1 : 0)
                  << " recenter_cells=" << tracker_recenter_cells
                  << " recenter_mode=" << tracker_recenter_mode_name(tracker_recenter_mode)
                  << std::endl;

        bool   constraint_violation = false;
        bool   gauge_instability = false;
        size_t failure_step = std::numeric_limits<size_t>::max();

        for (size_t n = 0; n <= cfg.steps; ++n) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(
                grid, tensorium::tests::make_cfl_control(cfg), cfg.padding);

            stepper.step(grid, dt, n);
            t += dt;
            advance_shift_puncture_tracker(grid, puncture_shift_tracker, dt);

            auto punctures_shift = make_tracker_sample(grid, puncture_shift_tracker);
            auto punctures_minima =
                sample_punctures_equatorial(grid, &puncture_minima_history, t,
                                            tracker_recenter_mode);
            if (punctures_minima.has_left && punctures_minima.has_right) {
                if (puncture_minima_history.has_last) {
                    puncture_minima_history.prev = puncture_minima_history.last;
                    puncture_minima_history.t_prev = puncture_minima_history.t_last;
                    puncture_minima_history.has_prev = true;
                }
                puncture_minima_history.last = punctures_minima;
                puncture_minima_history.t_last = t;
                puncture_minima_history.has_last = true;
            }

            bool   recentered = false;
            double drift_left = std::numeric_limits<double>::quiet_NaN();
            double drift_right = std::numeric_limits<double>::quiet_NaN();
            bool   have_drift = false;
            auto compute_tracker_drift = [&](double &dl, double &dr) -> bool {
                if (!(punctures_shift.has_left && punctures_shift.has_right &&
                      punctures_minima.has_left && punctures_minima.has_right))
                    return false;
                dl = std::hypot(punctures_shift.x_left - punctures_minima.x_left,
                                punctures_shift.y_left - punctures_minima.y_left);
                dr = std::hypot(punctures_shift.x_right - punctures_minima.x_right,
                                punctures_shift.y_right - punctures_minima.y_right);
                return std::isfinite(dl) && std::isfinite(dr);
            };
            have_drift = compute_tracker_drift(drift_left, drift_right);

            if (tracker_recenter_on_drift && have_drift &&
                (drift_left > tracker_recenter_radius || drift_right > tracker_recenter_radius)) {
                puncture_shift_tracker.p1[0] = punctures_minima.x_left;
                puncture_shift_tracker.p1[1] = punctures_minima.y_left;
                puncture_shift_tracker.p2[0] = punctures_minima.x_right;
                puncture_shift_tracker.p2[1] = punctures_minima.y_right;
                clamp_tracker_to_domain(grid, puncture_shift_tracker.p1);
                clamp_tracker_to_domain(grid, puncture_shift_tracker.p2);
                std::array<double, 3> b1{0.0, 0.0, 0.0};
                std::array<double, 3> b2{0.0, 0.0, 0.0};
                if (sample_vector3_tracker_interp(grid, grid.beta, puncture_shift_tracker.p1, b1))
                    puncture_shift_tracker.beta1_prev = b1;
                if (sample_vector3_tracker_interp(grid, grid.beta, puncture_shift_tracker.p2, b2))
                    puncture_shift_tracker.beta2_prev = b2;
                recentered = true;
                punctures_shift = make_tracker_sample(grid, puncture_shift_tracker);
                have_drift = compute_tracker_drift(drift_left, drift_right);
            }

            if (have_drift &&
                (drift_left > tracker_drift_warn_radius || drift_right > tracker_drift_warn_radius) &&
                (n % log_stride == 0)) {
                std::cout << "[tracker][warn] step=" << n << " drift_left=" << drift_left
                          << " drift_right=" << drift_right
                          << " warn_radius=" << tracker_drift_warn_radius
                          << " recentered=" << (recentered ? 1 : 0) << std::endl;
            }

            if (puncture_track.is_open()) {
                puncture_track << n << "," << t << ",";
                if (punctures_shift.has_left) {
                    puncture_track << punctures_shift.x_left << "," << punctures_shift.y_left
                                   << "," << punctures_shift.chi_left << ","
                                   << punctures_shift.alpha_left << ",";
                } else {
                    puncture_track << "nan,nan,nan,nan,";
                }
                if (punctures_shift.has_right) {
                    puncture_track << punctures_shift.x_right << "," << punctures_shift.y_right
                                   << "," << punctures_shift.chi_right << ","
                                   << punctures_shift.alpha_right << "\n";
                } else {
                    puncture_track << "nan,nan,nan,nan\n";
                }
            }
            if (puncture_track_minima.is_open()) {
                puncture_track_minima << n << "," << t << ",";
                if (punctures_minima.has_left) {
                    puncture_track_minima << punctures_minima.x_left << ","
                                          << punctures_minima.y_left << ","
                                          << punctures_minima.chi_left << ","
                                          << punctures_minima.alpha_left << ",";
                } else {
                    puncture_track_minima << "nan,nan,nan,nan,";
                }
                if (punctures_minima.has_right) {
                    puncture_track_minima << punctures_minima.x_right << ","
                                          << punctures_minima.y_right << ","
                                          << punctures_minima.chi_right << ","
                                          << punctures_minima.alpha_right << "\n";
                } else {
                    puncture_track_minima << "nan,nan,nan,nan\n";
                }
            }
            if (puncture_tracker_drift.is_open()) {
                puncture_tracker_drift << n << "," << t << ","
                                      << (have_drift ? drift_left
                                                     : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (have_drift ? drift_right
                                                     : std::numeric_limits<double>::quiet_NaN())
                                      << "," << (recentered ? 1 : 0) << ","
                                      << (punctures_shift.has_left ? punctures_shift.x_left
                                                                   : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_shift.has_left ? punctures_shift.y_left
                                                                   : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_shift.has_right ? punctures_shift.x_right
                                                                    : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_shift.has_right ? punctures_shift.y_right
                                                                    : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_minima.has_left ? punctures_minima.x_left
                                                                    : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_minima.has_left ? punctures_minima.y_left
                                                                    : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_minima.has_right ? punctures_minima.x_right
                                                                     : std::numeric_limits<double>::quiet_NaN())
                                      << ","
                                      << (punctures_minima.has_right ? punctures_minima.y_right
                                                                     : std::numeric_limits<double>::quiet_NaN())
                                      << "\n";
            }

            if (projection_stride > 0 && (n % projection_stride == 0))
                tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);

            const bool need_constraint_export = (n % constraint_export_stride == 0);
            const bool need_constraint_slice_export =
                export_constraint_slices && (n % constraint_slice_stride == 0);
            const bool need_constraint_log = (n % log_stride == 0);

            tensorium_RG::bssn::ConstraintMonitorStats stats;
            bool                                       have_stats = false;
            if (need_constraint_export || need_constraint_slice_export || need_constraint_log) {
                stats = collect_diagnostics();
                have_stats = true;
            }

            if (need_constraint_export && constraint_log.is_open() && have_stats) {
                constraint_log << n << "," << t << "," << dt << "," << stats.l2_theta << ","
                               << stats.l2_Z << "," << stats.l2_H << "," << stats.l2_M << ","
                               << stats.max_H << "," << stats.max_det_drift << ","
                               << stats.max_trace_A << "," << stats.samples << "\n";
            }
            if (need_constraint_slice_export && have_stats) {
                // Export full-domain constraints for visualization so slices do not appear
                // artificially clipped by the radial monitoring mask.
                compute_constraints_full_domain_for_slice();
                export_constraint_slice_csv(grid, constraint_scratch, n, "Output/viz");
            }

            if (need_constraint_log &&
                !log_diagnostics(n, have_stats ? &stats : nullptr)) {
                constraint_violation = true;
                failure_step = n;
                break;
            }

            if (!guard_gauge(n)) {
                gauge_instability = true;
                failure_step = n;
                break;
            }

            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n, cfg.steps, t);
        }

        if (constraint_violation) {
            std::ostringstream oss;
            oss << "Constraint growth exceeded stability limits at step " << failure_step;
            tensorium::tests::raise_failure(oss.str());
        }

        if (gauge_instability) {
            std::ostringstream oss;
            oss << "Gauge collapse detected (alpha or chi became singular) at step "
                << failure_step;
            tensorium::tests::raise_failure(oss.str());
        }
    });
REGISTER_TEST("bssn.viz.moving_puncture_interpolate",
              "Export CSV slices of moving punctures using interpolated initial data", []() {
                  ScopedEnvOverride force_interp("TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT",
                                                 "1");
                  const auto &tests = tensorium::tests::TestRegistry::instance().tests();
                  const auto  it = std::find_if(
                      tests.begin(), tests.end(), [](const tensorium::tests::TestCase &tc) {
                          return tc.name == "bssn.viz.moving_puncture";
                      });
                  if (it == tests.end()) {
                      tensorium::tests::raise_failure(
                          "Missing base test bssn.viz.moving_puncture");
                  }
                  it->func();
              });
REGISTER_TEST(
    "bssn.init.kerr_schild", "Print formatted BSSN fields after Kerr–Schild initialization", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 96;
        cfg.ny = 96;
        cfg.nz = 96;
        cfg.spacing = 0.3;
        cfg.ng = 4;
        cfg.steps = 60;
        cfg.cfl = 0.15;
        cfg.gauge_factor = 1.0;

        tensorium_RG::fd::set_fd_dx(cfg.spacing);

        tensorium::tests::Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing,
                                    cfg.spacing);

        grid.x0 = -0.5 * cfg.spacing * cfg.nx + 0.5 * cfg.spacing;
        grid.y0 = -0.5 * cfg.spacing * cfg.ny + 0.5 * cfg.spacing;
        grid.z0 = -0.5 * cfg.spacing * cfg.nz + 0.5 * cfg.spacing;

        tensorium_RG::init::kerr_schild_single(grid, 1.0, 0.9);

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

        const size_t                                ic = I0 + grid.dims.nx / 2;
        const size_t                                jc = J0 + grid.dims.ny / 2;
        const size_t                                kc = K0 + grid.dims.nz / 2;
        tensorium_RG::bssn::GaugeParameters<double> params;
        params.eta = 2.0;
        params.beta_B_coeff = 0.75;

        tensorium_RG::init::print_bssn_state_at(grid, ic, jc, kc, "Kerr–Schild single BH");
        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, cfg.padding);
        stepper.set_gauge_parameters(params);

        stepper.set_snapshot_callback([](const tensorium::tests::Grid &g, size_t step) {
            if (step % 1 == 0) {
                std::cout << ">> Exporting slice " << step << "..." << std::endl;
                export_slice_csv(g, step, "Output/viz");
            }
        });

        double t = 0.0;

        for (size_t n = 0; n <= cfg.steps; ++n) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(
                grid, tensorium::tests::make_cfl_control(cfg), cfg.padding);

            stepper.step(grid, dt, n);
            t += dt;

            std::printf("dt = %.4e\n", dt);
            std::printf("Step %zu / %zu (t=%.4f)\n", n, cfg.steps, t);
        }
        std::cout << "Kerr–Schild init test completed.\n";
    });
