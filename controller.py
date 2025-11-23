import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Controller update period (matches RaceCar.time_step)
DT = 0.1

# --- High-level steering mapping (heading error -> desired steering angle) ---
BASE_K_HEADING = 0.6     # base heading gain

STEER_LOOKAHEAD = 5      # look ahead along centerline for steering target

# --- Low-level steering PD (on steering angle delta) ---
Kp_delta = 1.8
Ki_delta = 0.0           # keep integral off for steering
Kd_delta = 0.9           # derivative for extra damping

# --- Low-level velocity PI (on speed v) ---
Kp_v = 0.7
Ki_v = 0.15
Kd_v = 0.0               # D not really needed on speed


def _init_pid_state() -> None:
    """Lazy init of PID state so it persists across calls."""
    if not hasattr(lower_controller, "delta_int"):
        lower_controller.delta_int = 0.0
        lower_controller.delta_prev_err = 0.0
        lower_controller.v_int = 0.0
        lower_controller.v_prev_err = 0.0


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller: maps (state, desired) -> [steering_rate, acceleration].

    state   = [sx, sy, delta, v, phi]
    desired = [delta_r, v_r]
    input   = [steering_rate, acceleration]
    """
    _init_pid_state()

    assert state.shape == (5,)
    assert desired.shape == (2,)
    assert parameters.shape == (11,)

    delta = state[2]
    v = state[3]

    delta_r = desired[0]
    v_r = desired[1]

    # ----- Steering PD (on steering angle delta) -----
    e_delta = np.arctan2(np.sin(delta_r - delta), np.cos(delta_r - delta))

    lower_controller.delta_int += e_delta * DT
    lower_controller.delta_int = np.clip(lower_controller.delta_int, -2.0, 2.0)

    delta_der = (e_delta - lower_controller.delta_prev_err) / DT
    lower_controller.delta_prev_err = e_delta

    steering_rate = (
        Kp_delta * e_delta
        + Ki_delta * lower_controller.delta_int
        + Kd_delta * delta_der
    )

    # ----- Velocity PI (on speed v) -----
    e_v = v_r - v

    lower_controller.v_int += e_v * DT
    lower_controller.v_int = np.clip(lower_controller.v_int, -50.0, 50.0)

    v_der = (e_v - lower_controller.v_prev_err) / DT
    lower_controller.v_prev_err = e_v

    acceleration = (
        Kp_v * e_v
        + Ki_v * lower_controller.v_int
        + Kd_v * v_der
    )

    # Clip using parameter limits
    steering_rate = np.clip(steering_rate, parameters[7], parameters[9])
    acceleration  = np.clip(acceleration,  parameters[8], parameters[10])

    return np.array([steering_rate, acceleration])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller:
      - choose reference steering angle delta_r
      - choose reference speed v_r (curvature-based, anticipatory)

    state = [sx, sy, delta, v, phi]
    """
    assert state.shape == (5,)
    assert parameters.shape == (11,)

    sx, sy = state[0], state[1]
    v      = state[3]
    phi    = state[4]

    delta_min = parameters[1]
    v_min     = parameters[2]
    delta_max = parameters[4]
    v_max     = parameters[5]

    centerline = racetrack.centerline
    N = centerline.shape[0]

    # ---------- 1) Find closest centerline point ----------
    car_pos = np.array([sx, sy])
    dists   = np.linalg.norm(centerline - car_pos, axis=1)
    idx     = int(np.argmin(dists))

    # ---------- 2) Compute curvature ahead of the car ----------
    # Use three points (prev, curv, next) to approximate curvature.
    idx_curv = (idx + 2) % N
    idx_prev = (idx_curv - 1) % N
    idx_next = (idx_curv + 1) % N

    p_prev = centerline[idx_prev]
    p0     = centerline[idx_curv]
    p_next = centerline[idx_next]

    a = np.linalg.norm(p0 - p_prev)
    b = np.linalg.norm(p_next - p0)
    c = np.linalg.norm(p_next - p_prev)

    if a > 1e-6 and b > 1e-6 and c > 1e-6:
        # 2 * area of triangle = |cross(p0-p_prev, p_next-p_prev)|
        area2 = abs(np.cross(p0 - p_prev, p_next - p_prev))
        curvature = 2.0 * area2 / (a * b * c)  # |kappa|
    else:
        curvature = 0.0

    kappa = curvature

    # ---------- 3) Curvature-based velocity reference ----------
    # Physics-ish: v_max(curvature) ~ sqrt(a_lat_max / |kappa|)
    v_straight   = min(v_max, 60.0)  # high speed on straights
    v_min_corner = 6.0               # crawl in the tightest corners
    a_lat_max    = 18.0              # "max lateral accel" (tuning knob)

    if kappa < 1e-6:
        v_curv = v_straight
    else:
        v_curv = np.sqrt(a_lat_max / kappa)

    # clip into [v_min_corner, v_straight]
    v_r = np.clip(v_curv, v_min_corner, v_straight)
    v_r = np.clip(v_r, v_min, v_max)

    # Build a curvature scale in [0,1] from this v_r for steering adaptation
    curv_scale = (v_r - v_min_corner) / (v_straight - v_min_corner)
    curv_scale = float(np.clip(curv_scale, 0.0, 1.0))

    # ---------- 4) Steering target & smoothed delta_r ----------
    idx_steer = (idx + STEER_LOOKAHEAD) % N
    target    = centerline[idx_steer]
    dx        = target[0] - sx
    dy        = target[1] - sy

    phi_d = np.arctan2(dy, dx)
    e_phi = np.arctan2(np.sin(phi_d - phi), np.cos(phi_d - phi))

    # Adaptive heading gain: smaller in sharp corners to reduce oscillation
    K_heading_eff = BASE_K_HEADING * (0.5 + 0.5 * curv_scale)
    raw_delta_r   = K_heading_eff * e_phi

    # soften demands at very low speed so we do not instantly saturate
    if abs(v) < 1.0:
        raw_delta_r = np.clip(raw_delta_r, -0.5, 0.5)

    # Adaptive smoothing: stronger smoothing (smaller alpha) in sharp corners
    # curv_scale ~ 1 (straight) -> alpha ~ 0.9
    # curv_scale ~ 0 (sharp)    -> alpha ~ 0.5
    alpha = 0.5 + 0.4 * curv_scale

    if not hasattr(controller, "delta_r_prev"):
        controller.delta_r_prev = raw_delta_r

    delta_r = (1.0 - alpha) * controller.delta_r_prev + alpha * raw_delta_r
    controller.delta_r_prev = delta_r

    # Clip to physical steering limits
    delta_r = np.clip(delta_r, delta_min, delta_max)

    # ---------- 5) Startup nudge ----------
    if v < 1.0:
        v_r = max(v_r, 8.0)

    return np.array([delta_r, v_r])
