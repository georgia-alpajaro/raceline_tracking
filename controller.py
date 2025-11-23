import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Controller update period (matches RaceCar.time_step)
DT = 0.1

# --- High-level steering mapping (heading error -> desired steering angle) ---
K_HEADING = 0.8        # slightly gentler than before
STEER_LOOKAHEAD = 5    # look further ahead so we see sharp turns earlier

# --- Low-level steering PD (on steering angle delta) ---
Kp_delta = 2.0
Ki_delta = 0.0         # keep integral off for steering (avoids windup / spinning)
Kd_delta = 0.5

# --- Low-level velocity PI (on speed v) ---
Kp_v = 0.7
Ki_v = 0.15
Kd_v = 0.0             # D not really needed on speed


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
      - choose reference speed v_r (curvature-based)

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

    # ---------- 2) Steering target: look ahead a few points ----------
    idx_steer = (idx + STEER_LOOKAHEAD) % N
    target    = centerline[idx_steer]
    dx        = target[0] - sx
    dy        = target[1] - sy

    phi_d = np.arctan2(dy, dx)
    e_phi = np.arctan2(np.sin(phi_d - phi), np.cos(phi_d - phi))

    delta_r = K_HEADING * e_phi

    # soften demands at very low speed so we do not instantly saturate
    if abs(v) < 1.0:
        delta_r = np.clip(delta_r, -0.5, 0.5)

    delta_r = np.clip(delta_r, delta_min, delta_max)

    # ---------- 3) Curvature-based velocity reference ----------
    idx_prev = (idx - 1) % N
    idx_next = (idx + 1) % N

    seg_prev = centerline[idx]      - centerline[idx_prev]
    seg_next = centerline[idx_next] - centerline[idx]

    phi_prev = np.arctan2(seg_prev[1], seg_prev[0])
    phi_next = np.arctan2(seg_next[1], seg_next[0])

    dphi = np.arctan2(np.sin(phi_next - phi_prev), np.cos(phi_next - phi_prev))
    dphi = abs(dphi)   # curvature proxy

    # speeds
    v_straight   = min(v_max, 50.0)  # high speed on straights
    v_min_corner = 10.0              # low speed in very tight corners

    # NEW: exponential mapping makes very sharp corners much slower
    # dphi ≈ 0   -> curv_scale ≈ 1  (full straight speed)
    # dphi ≈ 0.3 -> curv_scale ≈ exp(-6*0.09) ≈ 0.58
    # dphi ≈ 0.7 -> curv_scale ≈ exp(-6*0.49) ≈ 0.05 (big slow-down)
    curv_scale = np.exp(-6.0 * dphi * dphi)

    v_r = v_min_corner + (v_straight - v_min_corner) * curv_scale

    # clip within physical limits
    v_r = np.clip(v_r, v_min, v_max)

    # give a bit of push from rest so it starts moving
    if v < 1.0:
        v_r = max(v_r, 8.0)

    return np.array([delta_r, v_r])
