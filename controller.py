import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack  # small fix: RaceTrack lives in racetrack.py

# This should match RaceCar.time_step in racecar.py
CONTROL_DT = 0.1  # seconds

# Gains for the lower-level controller
K_DELTA = 1.5     # steering P gain (on steering angle)
K_V = 1.0         # velocity P gain (on speed)


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Lower-level controller C_v and steering-rate controller.
    Inputs:
      state     = [sx, sy, delta, v, phi]
      desired   = [delta_r, v_r]
      parameters = RaceCar.parameters (see racecar.py)
    Output:
      u = [steering_rate, acceleration]
    """
    assert state.shape == (5,)
    assert desired.shape == (2,)
    assert parameters.shape == (11,)

    # Unpack state
    delta = state[2]  # current steering angle
    v = state[3]      # current velocity

    # Desired references
    delta_r = desired[0]
    v_r = desired[1]

    # Wrap steering error to [-pi, pi] to avoid weird jumps
    e_delta = np.arctan2(np.sin(delta_r - delta), np.cos(delta_r - delta))
    e_v = v_r - v

    # Simple proportional controllers
    steering_rate = K_DELTA * e_delta           # v_delta
    acceleration = K_V * e_v                    # a

    # Input limits are enforced again in RaceCar.normalize_system, but we can
    # optionally clip here for safety using parameters:
    # parameters[7:9] = [min_steer_rate, min_acc]
    # parameters[9:11] = [max_steer_rate, max_acc]
    steering_rate = np.clip(steering_rate, parameters[7], parameters[9])
    acceleration = np.clip(acceleration, parameters[8], parameters[10])

    return np.array([steering_rate, acceleration])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Upper-level controller S1 + velocity reference generator.
    Computes:
      delta_r: reference steering angle (rad)
      v_r:     reference speed (m/s)

    Based on:
      - nearest centerline point
      - next centerline point for heading / velocity
    """
    assert state.shape == (5,)
    assert parameters.shape == (11,)

    # Unpack state (see racecar.py and racetrack.py)
    sx, sy = state[0], state[1]
    v = state[3]
    phi = state[4]

    # Car / limits from parameters
    lwb = parameters[0]        # wheelbase
    delta_min = parameters[1]
    delta_max = parameters[4]
    v_min = parameters[2]
    v_max = parameters[5]

    # === 1) Find closest point on centerline ===
    centerline = racetrack.centerline  # shape (N, 2)
    diffs = centerline - np.array([sx, sy])
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)

    # Use the *next* point as our local target (like i+1 in the assignment).
    # Wrap around to keep it on the track loop.
    idx_next = (idx + 1) % centerline.shape[0]

    target = centerline[idx_next]
    dx = target[0] - sx
    dy = target[1] - sy

    # === 2) Desired heading and steering angle reference (Q2) ===
    # Desired heading from current position to next reference point
    phi_d = np.arctan2(dy, dx)

    # Heading error, wrapped to [-pi, pi]
    e_phi = np.arctan2(np.sin(phi_d - phi), np.cos(phi_d - phi))

    # Use linearized relation: phi_dot ≈ (v / lwb) * delta   (tan(delta) ≈ delta)
    # Euler: phi[i+1] ≈ phi[i] + dt * (v / lwb) * delta_r
    # Set phi[i+1] ≈ phi_d => delta_r ≈ (lwb / (v * dt)) * (phi_d - phi)
    if abs(v) < 0.1:
        # Avoid divide-by-zero at very low speed; just point wheels straight-ish
        delta_r = 0.0
    else:
        delta_r = (lwb / (v * CONTROL_DT)) * e_phi

    # Clamp steering angle to physical limits
    delta_r = np.clip(delta_r, delta_min, delta_max)

    # === 3) Velocity reference (Q3) ===
    # Use speed = distance / time between successive centerline points.
    # Approximate time step for the reference as CONTROL_DT.
    segment = centerline[idx_next] - centerline[idx]
    seg_dist = np.linalg.norm(segment)

    if CONTROL_DT > 0:
        v_r = seg_dist / CONTROL_DT
    else:
        v_r = v_max  # fallback, should not happen

    # Clamp to vehicle velocity limits
    v_r = np.clip(v_r, v_min, v_max)

    # You could also enforce a minimum forward speed if desired:
    # v_r = max(v_r, 5.0)

    return np.array([delta_r, v_r])
