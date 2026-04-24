from __future__ import annotations

import contextlib
import math
import os

from . import tensorium as _C

Z4cGrid = _C.z4c.Z4cGrid
GaugeParameters = _C.z4c.GaugeParameters
Z4cRKStepper = _C.z4c.Z4cRKStepper

XX = _C.z4c.XX
XY = _C.z4c.XY
XZ = _C.z4c.XZ
YY = _C.z4c.YY
YZ = _C.z4c.YZ
ZZ = _C.z4c.ZZ


def has_twopunctures_c() -> bool:
    """Return True if this extension was compiled with TwoPuncturesC support."""
    return bool(_C.z4c.has_twopunctures_c())


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def apply_radiative_halos(grid: Z4cGrid) -> None:
    """Apply the BoundaryRadiative halo operator to all grid fields."""
    _C.z4c.apply_radiative_halos(grid)


def project_z4c_state(grid: Z4cGrid) -> None:
    """Project Z4c algebraic constraints on the current state."""
    _C.z4c.project_z4c_state(grid)


project_state = project_z4c_state


def zero_z4c_fields(grid: Z4cGrid) -> None:
    """Set Theta/Z fields to zero."""
    _C.z4c.zero_z4c_fields(grid)


def set_spatial_derivative_order(order: int) -> None:
    """Set spatial finite-difference order (4 or 6)."""
    _C.z4c.set_spatial_derivative_order(int(order))


def spatial_derivative_order() -> int:
    """Current spatial finite-difference order."""
    return int(_C.z4c.spatial_derivative_order())


def set_fd_dx(dx: float) -> None:
    """Set FD spacing used by derivative helpers."""
    _C.z4c.set_fd_dx(float(dx))


def compute_dt_cfl(grid: Z4cGrid, cfl: float, gauge_speed: float = 1.0, padding: int = 4) -> float:
    """Compute stable timestep estimate from CFL control."""
    return float(_C.z4c.compute_dt_cfl(grid, float(cfl), float(gauge_speed), int(padding)))


def set_boundary_faces(
    rhs_ix1: bool, rhs_ox1: bool, rhs_ix2: bool, rhs_ox2: bool, rhs_ix3: bool, rhs_ox3: bool,
    rf_ix1: bool, rf_ox1: bool, rf_ix2: bool, rf_ox2: bool, rf_ix3: bool, rf_ox3: bool
) -> None:
    """Configure Sommerfeld and reflective masks on each boundary face."""
    _C.z4c.set_boundary_faces(
        bool(rhs_ix1), bool(rhs_ox1), bool(rhs_ix2), bool(rhs_ox2), bool(rhs_ix3), bool(rhs_ox3),
        bool(rf_ix1), bool(rf_ox1), bool(rf_ix2), bool(rf_ox2), bool(rf_ix3), bool(rf_ox3)
    )


def minkowski(grid: Z4cGrid, M: float = 1.0, xc: float = 0.0, yc: float = 0.0, zc: float = 0.0,
              r_floor: float = 1e-6) -> None:
    """Initialize Minkowski data on the Z4c grid."""
    _C.z4c.minkowski(grid, M, xc, yc, zc, r_floor)


def schwarzschild_isotropic(grid: Z4cGrid, M: float, xc: float = 0.0, yc: float = 0.0,
                            zc: float = 0.0, r_floor: float = 1e-6) -> None:
    """Initialize isotropic Schwarzschild puncture data."""
    _C.z4c.schwarzschild_isotropic(grid, M, xc, yc, zc, r_floor)


def kerr_schild_single(grid: Z4cGrid, M: float, a: float, xc: float = 0.0, yc: float = 0.0,
                       zc: float = 0.0, r_floor: float = 1e-6) -> None:
    """Initialize a single Kerr-Schild black hole."""
    _C.z4c.kerr_schild_single(grid, M, a, xc, yc, zc, r_floor)


def binary_bowen_york_puncture_init(grid: Z4cGrid, m1: float, x1: float, y1: float, z1: float,
                                    P1, S1, m2: float, x2: float, y2: float, z2: float, P2, S2,
                                    r_floor: float = 1e-6) -> None:
    """Initialize binary Bowen-York puncture data."""
    _C.z4c.binary_bowen_york_puncture_init(grid, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2,
                                            r_floor)


def binary_bowen_york_puncture_interpolated_init(
    grid: Z4cGrid, m1: float, x1: float, y1: float, z1: float, P1, S1, m2: float, x2: float,
    y2: float, z2: float, P2, S2, interp_seed_n: int = 64, r_floor: float = 1e-6
) -> None:
    """Initialize interpolated binary Bowen-York puncture data."""
    _C.z4c.binary_bowen_york_puncture_interpolated_init(
        grid, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2, interp_seed_n, r_floor
    )


def binary_bowen_york_puncture_twopunctures_c_init(
    grid: Z4cGrid, m1: float, x1: float, y1: float, z1: float, P1, S1, m2: float, x2: float,
    y2: float, z2: float, P2, S2, interp_seed_n: int = 64, r_floor: float = 1e-6,
    verbose: bool | None = None, npoints_A: int | None = None, npoints_B: int | None = None,
    npoints_phi: int | None = None, newton_tol: float | None = None,
    newton_maxit: int | None = None, tp_epsilon: float | None = None,
    tp_threads: int | None = None
) -> None:
    """
    Initialize interpolated Bowen-York puncture data using the TwoPuncturesC backend.

    Raises:
      RuntimeError: if the extension was not compiled with TwoPuncturesC support.
    """
    if not has_twopunctures_c():
        raise RuntimeError("TwoPuncturesC support is not enabled in this build")

    env_updates: dict[str, str] = {}
    if verbose is not None:
        env_updates["TENSORIUM_TWOPUNCTURES_VERBOSE"] = "1" if verbose else "0"
    if npoints_A is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_A"] = str(int(npoints_A))
    if npoints_B is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_B"] = str(int(npoints_B))
    if npoints_phi is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_PHI"] = str(int(npoints_phi))
    if newton_tol is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_TOL"] = str(float(newton_tol))
    if newton_maxit is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_MAXIT"] = str(int(newton_maxit))
    if tp_epsilon is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_EPSILON"] = str(float(tp_epsilon))
    if tp_threads is not None:
        env_updates["TENSORIUM_MOVING_PUNCTURE_TP_THREADS"] = str(int(tp_threads))

    with _temporary_env(env_updates):
        _C.z4c.binary_bowen_york_puncture_twopunctures_c_init(
            grid, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2, interp_seed_n, r_floor
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw) != 0
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip()
    if "," in normalized:
        normalized = normalized.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return default


def _suggest_circular_momentum(m1: float, m2: float, separation: float) -> float | None:
    if m1 <= 0.0 or m2 <= 0.0 or separation <= 0.0:
        return None
    d = 2.0 * separation
    if d <= 0.0:
        return None
    return 0.295 / math.sqrt(d)


def run_moving_puncture_interpolate_from_env() -> tuple[Z4cGrid, float]:
    """
    Run a moving-puncture evolution from environment variables using the Python binding.

    Returns:
      (grid, t_final)
    """
    n = max(_env_int("TENSORIUM_MOVING_PUNCTURE_GRID_N", 96), 9)
    box_length = _env_float("TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH", 62.4)
    spacing = _env_float("TENSORIUM_MOVING_PUNCTURE_SPACING", box_length / float(n))
    ng = _env_int("TENSORIUM_MOVING_PUNCTURE_NG", 6)
    padding = max(_env_int("TENSORIUM_MOVING_PUNCTURE_PADDING", 0), 0)
    steps = max(_env_int("TENSORIUM_MOVING_PUNCTURE_STEPS", 6000), 1)
    cfl = max(_env_float("TENSORIUM_MOVING_PUNCTURE_CFL", 0.10), 1e-8)
    gauge_speed = max(_env_float("TENSORIUM_MOVING_PUNCTURE_GAUGE_SPEED", 1.0), 1e-8)

    set_spatial_derivative_order(_env_int("TENSORIUM_MOVING_PUNCTURE_SPATIAL_ORDER", 4))
    set_fd_dx(spacing)

    grid = Z4cGrid(n, n, n, ng, spacing, spacing, spacing)
    grid.set_origin(
        -0.5 * spacing * n + 0.5 * spacing,
        -0.5 * spacing * n + 0.5 * spacing,
        -0.5 * spacing * n + 0.5 * spacing,
    )

    m1 = _env_float("TENSORIUM_MOVING_PUNCTURE_MASS1", _env_float("TENSORIUM_MOVING_PUNCTURE_MASS", 0.48847892320123))
    m2 = _env_float("TENSORIUM_MOVING_PUNCTURE_MASS2", _env_float("TENSORIUM_MOVING_PUNCTURE_MASS", 0.48847892320123))
    separation = _env_float("TENSORIUM_MOVING_PUNCTURE_SEPARATION", 6.10679)
    momentum = _env_float("TENSORIUM_MOVING_PUNCTURE_MOMENTUM", 0.0841746)
    radial_momentum = _env_float("TENSORIUM_MOVING_PUNCTURE_RADIAL_MOMENTUM", 0.000510846)

    if _env_bool("TENSORIUM_MOVING_PUNCTURE_AUTO_CIRCULAR", False):
        p_circ = _suggest_circular_momentum(m1, m2, separation)
        if p_circ is not None:
            momentum = p_circ

    use_interpolated = _env_bool("TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT", True)
    interp_seed_n = max(_env_int("TENSORIUM_MOVING_PUNCTURE_INTERP_SEED_N", 64), 24)

    if use_interpolated:
        binary_bowen_york_puncture_twopunctures_c_init(
            grid,
            m1, -separation, 0.0, 0.0, [radial_momentum, -momentum, 0.0], [0.0, 0.0, 0.0],
            m2, separation, 0.0, 0.0, [-radial_momentum, momentum, 0.0], [0.0, 0.0, 0.0],
            interp_seed_n=interp_seed_n,
        )
    else:
        binary_bowen_york_puncture_init(
            grid,
            m1, 0.0, separation, 0.0, [-momentum, -radial_momentum, 0.0], [0.0, 0.0, 0.0],
            m2, 0.0, -separation, 0.0, [momentum, radial_momentum, 0.0], [0.0, 0.0, 0.0],
        )

    project_z4c_state(grid)
    zero_z4c_fields(grid)

    rhs_ix1 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX1", True)
    rhs_ox1 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX1", True)
    rhs_ix2 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX2", True)
    rhs_ox2 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX2", True)
    rhs_ix3 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX3", True)
    rhs_ox3 = _env_bool("TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX3", True)
    rf_ix1 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX1", False)
    rf_ox1 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX1", False)
    rf_ix2 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX2", False)
    rf_ox2 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX2", False)
    rf_ix3 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX3", False)
    rf_ox3 = _env_bool("TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX3", False)
    allow_reflective = _env_bool("TENSORIUM_MOVING_PUNCTURE_ALLOW_REFLECTIVE_BC", False)

    face_data = [
        ("ix1", rhs_ix1, rf_ix1),
        ("ox1", rhs_ox1, rf_ox1),
        ("ix2", rhs_ix2, rf_ix2),
        ("ox2", rhs_ox2, rf_ox2),
        ("ix3", rhs_ix3, rf_ix3),
        ("ox3", rhs_ox3, rf_ox3),
    ]
    for name, rhs_enabled, reflective in face_data:
        if reflective and not allow_reflective:
            raise RuntimeError(
                f"Reflective moving-puncture boundary requested on {name}. "
                "Set TENSORIUM_MOVING_PUNCTURE_ALLOW_REFLECTIVE_BC=1 to opt in explicitly."
            )

    if rf_ix1:
        rhs_ix1 = False
    if rf_ox1:
        rhs_ox1 = False
    if rf_ix2:
        rhs_ix2 = False
    if rf_ox2:
        rhs_ox2 = False
    if rf_ix3:
        rhs_ix3 = False
    if rf_ox3:
        rhs_ox3 = False

    set_boundary_faces(
        rhs_ix1, rhs_ox1, rhs_ix2, rhs_ox2, rhs_ix3, rhs_ox3,
        rf_ix1, rf_ox1, rf_ix2, rf_ox2, rf_ix3, rf_ox3,
    )

    params = GaugeParameters()
    params.eta = _env_float("TENSORIUM_MOVING_PUNCTURE_SHIFT_ETA", 1.0)
    params.beta_B_coeff = 0.75
    params.use_direct_shift_rhs = False
    params.use_shift_advection = False
    params.shift_Gamma = _env_float("TENSORIUM_MOVING_PUNCTURE_SHIFT_GAMMA", 0.75)
    params.shift_advect = 0.0
    params.shift_eta = params.eta
    params.lapse_oplog = 2.0
    params.lapse_advect = 1.0
    params.kappa1 = _env_float("TENSORIUM_MOVING_PUNCTURE_KAPPA1", 0.1)
    params.kappa2 = _env_float("TENSORIUM_MOVING_PUNCTURE_KAPPA2", 0.0)
    params.kappa3 = _env_float("TENSORIUM_MOVING_PUNCTURE_KAPPA3", 1.0)
    params.kappa_z = _env_float("TENSORIUM_MOVING_PUNCTURE_KAPPA_Z", 1.0)
    params.evolve_Z = _env_bool("TENSORIUM_MOVING_PUNCTURE_EVOLVE_Z", False)
    params.covariant_z4 = _env_bool("TENSORIUM_MOVING_PUNCTURE_COVARIANT_Z4", True)
    params.chi_div_floor = _env_float("TENSORIUM_MOVING_PUNCTURE_CHI_DIV_FLOOR", 1e-5)
    params.use_theta_in_lapse = True
    params.ko_sigma = _env_float("TENSORIUM_MOVING_PUNCTURE_KO_SIGMA", 1.0)
    params.slow_start_lapse = False
    params.min_lapse_for_K = 1e-4
    params.max_K_squared = 1e4
    params.alpha_floor = _env_float("TENSORIUM_MOVING_PUNCTURE_ALPHA_FLOOR", 1e-4)
    params.chi_floor = _env_float("TENSORIUM_MOVING_PUNCTURE_CHI_FLOOR", 1e-4)
    params.gamma_damping_uses_metric = False
    params.apply_rhs_sommerfeld = True
    params.use_direct_shift_rhs = _env_bool(
        "TENSORIUM_MOVING_PUNCTURE_USE_DIRECT_SHIFT_RHS", params.use_direct_shift_rhs
    )
    params.use_shift_advection = _env_bool(
        "TENSORIUM_MOVING_PUNCTURE_USE_SHIFT_ADVECTION", params.use_shift_advection
    )
    params.shift_advect = _env_float("TENSORIUM_MOVING_PUNCTURE_SHIFT_ADVECT", params.shift_advect)
    params.lapse_advect = _env_float("TENSORIUM_MOVING_PUNCTURE_LAPSE_ADVECT", params.lapse_advect)
    params.beta_B_coeff = _env_float("TENSORIUM_MOVING_PUNCTURE_BETA_B_COEFF", params.beta_B_coeff)

    stepper = Z4cRKStepper(grid, padding)
    stepper.set_gauge_parameters(params)
    stepper.set_state_log_stride(max(_env_int("TENSORIUM_MOVING_PUNCTURE_STATE_LOG_STRIDE", 10), 1))

    t = 0.0
    for nstep in range(steps):
        dt = compute_dt_cfl(grid, cfl=cfl, gauge_speed=gauge_speed, padding=padding)
        stepper.step(grid, dt, nstep)
        t += dt

    return grid, t


__all__ = [
    "Z4cGrid",
    "GaugeParameters",
    "Z4cRKStepper",
    "XX",
    "XY",
    "XZ",
    "YY",
    "YZ",
    "ZZ",
    "has_twopunctures_c",
    "apply_radiative_halos",
    "project_z4c_state",
    "project_state",
    "zero_z4c_fields",
    "set_spatial_derivative_order",
    "spatial_derivative_order",
    "set_fd_dx",
    "compute_dt_cfl",
    "set_boundary_faces",
    "minkowski",
    "schwarzschild_isotropic",
    "kerr_schild_single",
    "binary_bowen_york_puncture_init",
    "binary_bowen_york_puncture_interpolated_init",
    "binary_bowen_york_puncture_twopunctures_c_init",
    "run_moving_puncture_interpolate_from_env",
]
