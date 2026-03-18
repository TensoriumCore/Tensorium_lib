#pragma once

/**
 * @file MPIBSSNRK4.hpp
 * @brief MPI-distributed RK4 time integrator for BSSN evolution.
 * @details
 * Extends the single-process BSSNRKStepper with MPI halo exchanges at each
 * Runge-Kutta stage. This enables distributed-memory parallelism across
 * multiple nodes.
 *
 * The stepper performs halo exchanges after each RK stage to ensure ghost
 * zones contain valid data before the next RHS evaluation.
 */

#include "MPI.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <functional>
#include <memory>

namespace tensorium::mpi {

/**
 * @brief MPI-aware RK4 stepper for distributed BSSN evolution.
 * @tparam T Floating-point type (float or double).
 *
 * This stepper manages:
 * - Domain decomposition via MPIDomain
 * - Halo exchange between RK stages
 * - Global CFL time step computation
 * - Distributed constraint monitoring
 */
template <typename T> class MPIBSSNRKStepper {
  public:
    using GridType = tensorium_RG::BSSNGridSoA<T>;
    using GaugeParams = tensorium_RG::bssn::GaugeParameters<T>;
    using InteriorRegion = tensorium_RG::bssn::InteriorRegion;

    /**
     * @brief Construct MPI stepper.
     * @param domain MPI domain decomposition.
     * @param prototype Prototype grid (used for workspace allocation).
     * @param padding Interior padding for stencils.
     */
    MPIBSSNRKStepper(const MPIDomain &domain, const GridType &prototype, size_t padding = 4)
        : domain_(domain),
          padding_(padding),
          reductions_(domain),
          boundary_(domain, prototype.alpha.st, domain.local_nx(), domain.local_ny(),
                    domain.local_nz()),
          stage_grid_(domain.local_nx(), domain.local_ny(), domain.local_nz(), domain.ng(),
                      static_cast<T>(domain.dx()), static_cast<T>(domain.dy()),
                      static_cast<T>(domain.dz())) {

        stage_grid_.x0 = static_cast<T>(domain.local_x0());
        stage_grid_.y0 = static_cast<T>(domain.local_y0());
        stage_grid_.z0 = static_cast<T>(domain.local_z0());

        for (auto &stage : stages_) {
            stage.allocate_like(prototype);
        }

        tensorium_RG::bssn::detail::allocate_like(prototype.alpha, theta_cache_);
        tensorium_RG::bssn::detail::allocate_like(prototype.alpha, h_constraint_cache_);
        for (int q = 0; q < 3; ++q) {
            tensorium_RG::bssn::detail::allocate_like(prototype.alpha, m_constraint_cache_[q]);
            tensorium_RG::bssn::detail::allocate_like(prototype.alpha, c_constraint_cache_[q]);
        }

        MPIBoundaryAdapter<T>::instance = &boundary_;
    }

    ~MPIBSSNRKStepper() {
        if (MPIBoundaryAdapter<T>::instance == &boundary_) {
            MPIBoundaryAdapter<T>::instance = nullptr;
        }
    }

    /// @brief Set gauge evolution parameters.
    void set_gauge_parameters(const GaugeParams &params) { gauge_params_ = params; }

    /// @brief Set constraint monitoring callback (called after each step on root).
    void set_constraint_callback(
        std::function<void(const GridType &, const GlobalConstraintStats &)> cb) {
        constraint_callback_ = std::move(cb);
    }

    /// @brief Set snapshot callback (called after each step on every rank).
    void set_snapshot_callback(std::function<void(const GridType &, size_t)> cb) {
        snapshot_callback_ = std::move(cb);
    }

    /// @brief Get the MPI domain.
    const MPIDomain &domain() const { return domain_; }

    /// @brief Get the reductions helper.
    const Reductions &reductions() const { return reductions_; }

    /**
     * @brief Compute global CFL-limited time step.
     * @param grid Current grid state.
     * @param cfl CFL factor (default 0.25).
     * @return Global time step satisfying CFL across all processes.
     */
    T compute_dt(const GridType &grid, T cfl = T(0.25)) {
        return compute_global_dt_cfl(grid, domain_, cfl, padding_);
    }

    /**
     * @brief Perform one RK4 time step with MPI synchronization.
     * @param grid Grid to evolve (modified in-place).
     * @param dt Time step size.
     * @param step_index Current step number (for diagnostics).
     */
    void step(GridType &grid, T dt, size_t step_index = 0) {
        static constexpr std::array<double, 4> gam0_ref = {0.0, 0.121098479554482,
                                                           -3.843833699660025, 0.546370891121863};
        static constexpr std::array<double, 4> gam1_ref = {1.0, 0.721781678111411,
                                                           2.121209265338722, 0.198653035682705};
        static constexpr std::array<double, 4> beta_ref = {1.193743905974738, 0.099279895495783,
                                                           1.131678018054042, 0.310665766509336};
        static constexpr std::array<double, 4> delta_ref = {1.0, 0.217683334308543,
                                                            1.065841341361089, 0.0};

        boundary_dt_ = dt;

        for (int stage = 0; stage < 4; ++stage) {
            if (stage == 0) {
                copy_state(grid, stage_grid_);
            } else {
                accumulate_state(grid, stage_grid_, T(delta_ref[stage]));
            }

            prepare_state_for_rhs(grid);

            auto stage_params = gauge_params_;
            stage_params.current_time = simulation_time_;
            stage_params.frozen_Z_is_synced = !stage_params.evolve_Z;
            evaluate_rhs(grid, stages_[stage], stage_params);

            apply_stage_update(grid, stage_grid_, stages_[stage], T(gam0_ref[stage]),
                               T(gam1_ref[stage]), T(beta_ref[stage]) * dt);

            apply_floors(grid);
        }

        simulation_time_ += dt;

        apply_floors(grid);
        const bool do_state_log = ((step_index + 1) % state_log_stride_ == 0);
        const bool needs_post_step_prepare = do_state_log || static_cast<bool>(snapshot_callback_);
        if (needs_post_step_prepare) {
            prepare_state_for_rhs(grid);
        } else {
            synchronize_z_from_gamma(grid);
        }

        if (constraint_callback_ && do_state_log) {
            auto stats = compute_constraints(grid);
            stats.allreduce(domain_);
            if (domain_.is_root()) {
                constraint_callback_(grid, stats);
            }
        }

        if (do_state_log) {
            log_diagnostics(grid, step_index);
        }

        if (snapshot_callback_) {
            snapshot_callback_(grid, step_index);
        }
    }

    /// @brief Set stride for state logging.
    void set_state_log_stride(size_t stride) { state_log_stride_ = std::max<size_t>(1, stride); }

  private:
    const MPIDomain &domain_;
    size_t           padding_;
    Reductions       reductions_;
    MPIBoundary<T>   boundary_;

    GridType                                stage_grid_;
    tensorium_RG::bssn::BSSNRHSWorkspace<T> stages_[4];
    tensorium_RG::Field3D<T>                theta_cache_;
    tensorium_RG::Field3D<T>                h_constraint_cache_;
    tensorium_RG::Field3D<T>                m_constraint_cache_[3];
    tensorium_RG::Field3D<T>                c_constraint_cache_[3];

    GaugeParams gauge_params_{};
    T           boundary_dt_ = T(0);
    T           simulation_time_ = T(0);
    size_t      state_log_stride_ = 10;

    std::function<void(const GridType &, const GlobalConstraintStats &)> constraint_callback_;
    std::function<void(const GridType &, size_t)>                        snapshot_callback_;

    static constexpr HaloFieldMask rhs_halo_fields(bool evolve_z) {
        (void)evolve_z;
        HaloFieldMask mask = HaloFieldAlpha | HaloFieldChi | HaloFieldK | HaloFieldTheta |
                             HaloFieldBeta | HaloFieldB | HaloFieldTildeGamma |
                             HaloFieldGammaTilde | HaloFieldATilde;
        return mask;
    }

    void exchange_halos(GridType &grid, HaloFieldMask mask = HaloFieldAll) {
        boundary_.exchange_halos(grid, mask);
    }

    static InteriorRegion total_region(const GridType &grid) {
        return InteriorRegion{0, grid.alpha.st.nx_tot, 0, grid.alpha.st.ny_tot,
                              0, grid.alpha.st.nz_tot};
    }

    static InteriorRegion ricci_region(const GridType &grid) {
        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        return InteriorRegion{std::min(I0 + size_t(3), I1), (I1 > 3) ? I1 - 3 : I0,
                              std::min(J0 + size_t(3), J1), (J1 > 3) ? J1 - 3 : J0,
                              std::min(K0 + size_t(3), K1), (K1 > 3) ? K1 - 3 : K0};
    }

    static InteriorRegion shrink_region(const InteriorRegion &region, size_t padding) {
        return InteriorRegion{std::min(region.i0 + padding, region.i1),
                              (region.i1 > padding) ? region.i1 - padding : region.i0,
                              std::min(region.j0 + padding, region.j1),
                              (region.j1 > padding) ? region.j1 - padding : region.j0,
                              std::min(region.k0 + padding, region.k1),
                              (region.k1 > padding) ? region.k1 - padding : region.k0};
    }

    static InteriorRegion intersect_region(const InteriorRegion &lhs, const InteriorRegion &rhs) {
        return InteriorRegion{std::max(lhs.i0, rhs.i0), std::min(lhs.i1, rhs.i1),
                              std::max(lhs.j0, rhs.j0), std::min(lhs.j1, rhs.j1),
                              std::max(lhs.k0, rhs.k0), std::min(lhs.k1, rhs.k1)};
    }

    size_t physical_boundary_padding(bool enabled) const noexcept {
        (void)enabled;
        return size_t(0);
    }

    auto interior_padding_scope() const {
        return tensorium_RG::bssn::ScopedInteriorPaddingOverride(
            physical_boundary_padding(domain_.is_boundary_x_minus()),
            physical_boundary_padding(domain_.is_boundary_x_plus()),
            physical_boundary_padding(domain_.is_boundary_y_minus()),
            physical_boundary_padding(domain_.is_boundary_y_plus()),
            physical_boundary_padding(domain_.is_boundary_z_minus()),
            physical_boundary_padding(domain_.is_boundary_z_plus()));
    }

    template <typename Fn>
    static void for_each_shell_region(const InteriorRegion &outer, const InteriorRegion &inner,
                                      Fn &&fn) {
        if (outer.empty()) {
            return;
        }

        const InteriorRegion core = intersect_region(outer, inner);
        if (core.empty()) {
            fn(outer);
            return;
        }

        auto emit = [&](size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
            const InteriorRegion region{i0, i1, j0, j1, k0, k1};
            if (!region.empty()) {
                fn(region);
            }
        };

        emit(outer.i0, core.i0, outer.j0, outer.j1, outer.k0, outer.k1);
        emit(core.i1, outer.i1, outer.j0, outer.j1, outer.k0, outer.k1);
        emit(core.i0, core.i1, outer.j0, core.j0, outer.k0, outer.k1);
        emit(core.i0, core.i1, core.j1, outer.j1, outer.k0, outer.k1);
        emit(core.i0, core.i1, core.j0, core.j1, outer.k0, core.k0);
        emit(core.i0, core.i1, core.j0, core.j1, core.k1, outer.k1);
    }

    void prepare_state_for_rhs(GridType &grid) {
        enforce_floors(grid);
        tensorium_RG::bssn::BoundaryRadiative::set_characteristic(1.0, double(boundary_dt_));

        const InteriorRegion algebraic_outer = total_region(grid);
        const InteriorRegion algebraic_core = shrink_region(algebraic_outer, domain_.ng());
        const InteriorRegion ricci_outer = ricci_region(grid);
        const InteriorRegion ricci_core = shrink_region(ricci_outer, domain_.ng());

        typename MPIBoundary<T>::ExchangeState exchange_state;
        const HaloFieldMask pre_rhs_fields = rhs_halo_fields(gauge_params_.evolve_Z);
        boundary_.exchange_halos_start(grid, pre_rhs_fields, exchange_state);

        tensorium_RG::bssn::enforce_algebraic_constraints_region(
            grid, algebraic_core.i0, algebraic_core.i1, algebraic_core.j0, algebraic_core.j1,
            algebraic_core.k0, algebraic_core.k1);
        tensorium_RG::bssn::compute_ricci_bssn_region(grid, grid.Ricci, ricci_core.i0,
                                                      ricci_core.i1, ricci_core.j0, ricci_core.j1,
                                                      ricci_core.k0, ricci_core.k1, false);

        boundary_.exchange_halos_finish(grid, pre_rhs_fields, exchange_state);
        for_each_shell_region(algebraic_outer, algebraic_core, [&](const InteriorRegion &region) {
            enforce_floors_region(grid, region);
        });
        for_each_shell_region(algebraic_outer, algebraic_core, [&](const InteriorRegion &region) {
            tensorium_RG::bssn::enforce_algebraic_constraints_region(
                grid, region.i0, region.i1, region.j0, region.j1, region.k0, region.k1);
        });
        for_each_shell_region(ricci_outer, ricci_core, [&](const InteriorRegion &region) {
            tensorium_RG::bssn::compute_ricci_bssn_region(grid, grid.Ricci, region.i0, region.i1,
                                                          region.j0, region.j1, region.k0,
                                                          region.k1, false);
        });

        synchronize_z_from_gamma(grid);
    }

    void enforce_floors(GridType &grid) { enforce_floors_region(grid, total_region(grid)); }

    void enforce_floors_region(GridType &grid, const InteriorRegion &region) {
        if (region.empty() ||
            (gauge_params_.alpha_floor <= T(0) && gauge_params_.chi_floor <= T(0))) {
            return;
        }
        T      *alpha = grid.alpha.ptr();
        T      *chi = grid.chi.ptr();
        const T alpha_floor = gauge_params_.alpha_floor;
        const T chi_floor = gauge_params_.chi_floor;

#pragma omp parallel for collapse(3)
        for (size_t i = region.i0; i < region.i1; ++i)
            for (size_t j = region.j0; j < region.j1; ++j) {
                for (size_t k = region.k0; k < region.k1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    if (alpha_floor > T(0) && alpha[idx] < alpha_floor) {
                        alpha[idx] = alpha_floor;
                    }
                    if (chi_floor > T(0) && chi[idx] < chi_floor) {
                        chi[idx] = chi_floor;
                    }
                }
            }
    }

    void apply_floors(GridType &grid) {
        if (gauge_params_.alpha_floor > T(0)) {
            apply_smooth_floor(grid.alpha, gauge_params_.alpha_floor);
        }
        if (gauge_params_.chi_floor > T(0)) {
            apply_smooth_floor(grid.chi, gauge_params_.chi_floor);
        }
    }

    void apply_smooth_floor(tensorium_RG::Field3D<T> &field, T floor) {
        const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
        T           *ptr = field.ptr();
        const T      delta = T(1e-10);

#pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            T val = ptr[idx];
            ptr[idx] = T(0.5) * (val + floor + std::sqrt((val - floor) * (val - floor) + delta));
        }
    }

    void synchronize_z_from_gamma(GridType &grid) {
        using namespace tensorium_RG::fd;
        const double    inv_60dx = 1.0 / (60.0 * grid.dx);
        const double    inv_60dy = 1.0 / (60.0 * grid.dy);
        const double    inv_60dz = 1.0 / (60.0 * grid.dz);
        const ptrdiff_t sx = grid.alpha.st.sx;
        const ptrdiff_t sy = grid.alpha.st.sy;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i) {
            for (size_t j = J0; j < J1; ++j) {
                for (size_t k = K0; k < K1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    const T      chi = grid.chi.ptr()[idx];
                    const T      chi_guarded =
                        tensorium_RG::bssn::guard_chi_div(chi, gauge_params_.chi_div_floor);

                    const T *p_xx = grid.gamma_tilde_inv[tensorium_RG::XX].ptr() + idx;
                    const T *p_xy = grid.gamma_tilde_inv[tensorium_RG::XY].ptr() + idx;
                    const T *p_xz = grid.gamma_tilde_inv[tensorium_RG::XZ].ptr() + idx;
                    const T *p_yy = grid.gamma_tilde_inv[tensorium_RG::YY].ptr() + idx;
                    const T *p_yz = grid.gamma_tilde_inv[tensorium_RG::YZ].ptr() + idx;
                    const T *p_zz = grid.gamma_tilde_inv[tensorium_RG::ZZ].ptr() + idx;

                    T z_over_chi[3] = {T(0), T(0), T(0)};
                    tensorium_RG::bssn::recover_z_over_chi_from_gamma_ptr(
                        grid.tildeGamma[0].ptr() + idx, grid.tildeGamma[1].ptr() + idx,
                        grid.tildeGamma[2].ptr() + idx, p_xx, p_xy, p_xz, p_yy, p_yz, p_zz, sx,
                        sy, inv_60dx, inv_60dy, inv_60dz, z_over_chi);

                    grid.Z[0].ptr()[idx] = chi_guarded * z_over_chi[0];
                    grid.Z[1].ptr()[idx] = chi_guarded * z_over_chi[1];
                    grid.Z[2].ptr()[idx] = chi_guarded * z_over_chi[2];
                }
            }
        }
    }

    void evaluate_rhs(const GridType &grid, tensorium_RG::bssn::BSSNRHSWorkspace<T> &rhs,
                      const GaugeParams &params) {
        auto padding_scope = interior_padding_scope();
        tensorium_RG::bssn::update_ko_scale(grid, boundary_dt_);
        tensorium_RG::bssn::evaluate_rhs_sweep_core(
            grid, rhs.alpha, rhs.chi, rhs.K, rhs.Theta, rhs.beta, rhs.B, rhs.gamma_tilde,
            rhs.A_tilde, rhs.tildeGamma, rhs.Z, params, size_t(0), &theta_cache_);
        apply_rhs_sommerfeld(grid, rhs);
        tensorium_RG::bssn::recompose_rhs_K_from_khat_core(grid, rhs.K, rhs.Theta, size_t(0));
    }

    void apply_rhs_sommerfeld(const GridType &grid, tensorium_RG::bssn::BSSNRHSWorkspace<T> &rhs) {
        if (!gauge_params_.apply_rhs_sommerfeld) {
            return;
        }
        tensorium_RG::bssn::RHSBoundaryFaceMask mask{};
        mask.face[0][0] = domain_.is_boundary_x_minus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(0, false) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(0, false);
        mask.face[0][1] = domain_.is_boundary_x_plus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(0, true) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(0, true);
        mask.face[1][0] = domain_.is_boundary_y_minus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(1, false) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(1, false);
        mask.face[1][1] = domain_.is_boundary_y_plus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(1, true) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(1, true);
        mask.face[2][0] = domain_.is_boundary_z_minus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(2, false) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(2, false);
        mask.face[2][1] = domain_.is_boundary_z_plus() &&
                          tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_enabled(2, true) &&
                          !tensorium_RG::bssn::BoundaryRadiative::reflective_enabled(2, true);
        tensorium_RG::bssn::apply_z4c_rhs_boundary(grid, rhs, mask);
    }

    void copy_state(const GridType &src, GridType &dst) {
        dst.x0 = src.x0;
        dst.y0 = src.y0;
        dst.z0 = src.z0;
        {
            auto padding_scope = interior_padding_scope();
            copy_field_interior(src, src.alpha, dst.alpha);
            copy_field_interior(src, src.chi, dst.chi);
            copy_field_interior(src, src.K, dst.K);
            copy_field_interior(src, src.Theta, dst.Theta);
            for (int i = 0; i < 3; ++i) {
                copy_field_interior(src, src.beta[i], dst.beta[i]);
                copy_field_interior(src, src.B[i], dst.B[i]);
                copy_field_interior(src, src.tildeGamma[i], dst.tildeGamma[i]);
                copy_field_interior(src, src.Z[i], dst.Z[i]);
            }
            for (int s = 0; s < 6; ++s) {
                copy_field_interior(src, src.gamma_tilde[s], dst.gamma_tilde[s]);
                copy_field_interior(src, src.A_tilde[s], dst.A_tilde[s]);
            }
        }
    }

    void accumulate_state(const GridType &src, GridType &dst, T delta) {
        auto padding_scope = interior_padding_scope();
        accumulate_field_interior(src, src.alpha, dst.alpha, delta);
        accumulate_field_interior(src, src.chi, dst.chi, delta);
        accumulate_field_interior(src, src.K, dst.K, delta);
        accumulate_field_interior(src, src.Theta, dst.Theta, delta);
        for (int i = 0; i < 3; ++i) {
            accumulate_field_interior(src, src.beta[i], dst.beta[i], delta);
            accumulate_field_interior(src, src.B[i], dst.B[i], delta);
            accumulate_field_interior(src, src.tildeGamma[i], dst.tildeGamma[i], delta);
            accumulate_field_interior(src, src.Z[i], dst.Z[i], delta);
        }
        for (int s = 0; s < 6; ++s) {
            accumulate_field_interior(src, src.gamma_tilde[s], dst.gamma_tilde[s], delta);
            accumulate_field_interior(src, src.A_tilde[s], dst.A_tilde[s], delta);
        }
    }

    void apply_stage_update(GridType &u0, const GridType &u1,
                            const tensorium_RG::bssn::BSSNRHSWorkspace<T> &rhs, T gam0, T gam1,
                            T beta_dt) {
        {
            auto padding_scope = interior_padding_scope();
            update_field_interior(u0, u0.alpha, u1.alpha, rhs.alpha, gam0, gam1, beta_dt);
            update_field_interior(u0, u0.chi, u1.chi, rhs.chi, gam0, gam1, beta_dt);
            update_field_interior(u0, u0.K, u1.K, rhs.K, gam0, gam1, beta_dt);
            update_field_interior(u0, u0.Theta, u1.Theta, rhs.Theta, gam0, gam1, beta_dt);
            for (int i = 0; i < 3; ++i) {
                update_field_interior(u0, u0.beta[i], u1.beta[i], rhs.beta[i], gam0, gam1, beta_dt);
                update_field_interior(u0, u0.B[i], u1.B[i], rhs.B[i], gam0, gam1, beta_dt);
                update_field_interior(u0, u0.tildeGamma[i], u1.tildeGamma[i], rhs.tildeGamma[i],
                                      gam0, gam1, beta_dt);
                update_field_interior(u0, u0.Z[i], u1.Z[i], rhs.Z[i], gam0, gam1, beta_dt);
            }
            for (int s = 0; s < 6; ++s) {
                update_field_interior(u0, u0.gamma_tilde[s], u1.gamma_tilde[s], rhs.gamma_tilde[s],
                                      gam0, gam1, beta_dt);
                update_field_interior(u0, u0.A_tilde[s], u1.A_tilde[s], rhs.A_tilde[s], gam0, gam1,
                                      beta_dt);
            }
        }
        tensorium_RG::bssn::enforce_algebraic_constraints(u0);
    }

    void copy_field_interior(const GridType &grid, const tensorium_RG::Field3D<T> &src,
                             tensorium_RG::Field3D<T> &dst) {
        const T *in = src.ptr();
        T       *out = dst.ptr();
        tensorium_RG::bssn::for_each_interior_index_parallel(
            grid, padding_, [&](size_t, size_t, size_t, size_t idx) { out[idx] = in[idx]; });
    }

    void accumulate_field_interior(const GridType &grid, const tensorium_RG::Field3D<T> &src,
                                   tensorium_RG::Field3D<T> &dst, T scale) {
        const T *in = src.ptr();
        T       *out = dst.ptr();
        tensorium_RG::bssn::for_each_interior_index_parallel(
            grid, padding_,
            [&](size_t, size_t, size_t, size_t idx) { out[idx] += scale * in[idx]; });
    }

    void update_field_interior(const GridType &grid, tensorium_RG::Field3D<T> &u0,
                               const tensorium_RG::Field3D<T> &u1,
                               const tensorium_RG::Field3D<T> &rhs, T gam0, T gam1, T beta_dt) {
        T       *out = u0.ptr();
        const T *base = u1.ptr();
        const T *k = rhs.ptr();
        tensorium_RG::bssn::for_each_interior_index_parallel(
            grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
                out[idx] = gam0 * out[idx] + gam1 * base[idx] + beta_dt * k[idx];
            });
    }

    GlobalConstraintStats compute_constraints(const GridType &grid) {
        GlobalConstraintStats stats;
        auto                 &mutable_grid = const_cast<GridType &>(grid);
        tensorium_RG::bssn::compute_bssn_constraints(
            mutable_grid, grid.Ricci, h_constraint_cache_, m_constraint_cache_, c_constraint_cache_,
            0.0, std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);
        auto monitor =
            tensorium_RG::bssn::compute_constraint_monitor(grid, h_constraint_cache_, padding_);
        tensorium_RG::bssn::populate_constraint_norms(grid, m_constraint_cache_, monitor, padding_);

        const double samples = static_cast<double>(monitor.samples);
        stats.l2_hamiltonian = monitor.l2_H * monitor.l2_H * samples;
        stats.l2_momentum = monitor.l2_M * monitor.l2_M * samples;
        stats.l2_theta = monitor.l2_theta * monitor.l2_theta * samples;
        stats.l2_Z = monitor.l2_Z * monitor.l2_Z * samples;
        stats.max_hamiltonian = monitor.max_H;
        stats.max_det_drift = monitor.max_det_drift;
        stats.max_trace_A = monitor.max_trace_A;
        stats.total_samples = monitor.samples;

        return stats;
    }

    void log_diagnostics(const GridType &grid, size_t step_index) {
        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

        double min_alpha = 1e100, min_chi = 1e100;
        double max_theta = 0.0;
        size_t finite_alpha = 0;
        size_t finite_chi = 0;
        size_t finite_theta = 0;

        for (size_t i = I0; i < I1; ++i) {
            for (size_t j = J0; j < J1; ++j) {
                for (size_t k = K0; k < K1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    const double alpha = double(grid.alpha.ptr()[idx]);
                    const double chi = double(grid.chi.ptr()[idx]);
                    const double theta = std::abs(double(grid.Theta.ptr()[idx]));
                    if (std::isfinite(alpha)) {
                        min_alpha = std::min(min_alpha, alpha);
                        ++finite_alpha;
                    }
                    if (std::isfinite(chi)) {
                        min_chi = std::min(min_chi, chi);
                        ++finite_chi;
                    }
                    if (std::isfinite(theta)) {
                        max_theta = std::max(max_theta, theta);
                        ++finite_theta;
                    }
                }
            }
        }

        // Global reductions
        min_alpha = reductions_.allreduce_min(min_alpha);
        min_chi = reductions_.allreduce_min(min_chi);
        max_theta = reductions_.allreduce_max(max_theta);
        finite_alpha = reductions_.allreduce_sum(finite_alpha);
        finite_chi = reductions_.allreduce_sum(finite_chi);
        finite_theta = reductions_.allreduce_sum(finite_theta);

        if (finite_alpha == 0)
            min_alpha = std::numeric_limits<double>::quiet_NaN();
        if (finite_chi == 0)
            min_chi = std::numeric_limits<double>::quiet_NaN();
        if (finite_theta == 0)
            max_theta = std::numeric_limits<double>::quiet_NaN();

        if (domain_.is_root()) {
            printf("[MPI-BSSN %04zu] t=%.4f alpha_min=%.3e chi_min=%.3e Theta_max=%.3e\n",
                   step_index, double(simulation_time_), min_alpha, min_chi, max_theta);
            fflush(stdout);
        }
    }
};

} // namespace tensorium::mpi
