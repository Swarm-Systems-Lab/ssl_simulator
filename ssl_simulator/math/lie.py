"""
This module provides functions for working with 3D rotations and the Lie algebra so(3).
It includes utilities for generating rotation matrices, constructing orthonormal bases,
and computing exponential and logarithmic maps between SO(3) and its Lie algebra.

Notes:
    - The module assumes input vectors and matrices are NumPy arrays.
    - Some functions use approximations for small angles to improve numerical stability.
"""

__all__ = [
    "rot_3d_matrix",
    "gen_random_rotations",
    "orthonormal_vector_to",
    "construct_attitude_basis",
    "rotation_matrix_from_vector",
    "rotation_angle_from_matrix",
    "so3_hat",
    "so3_vee",
    "so3_exp_map",
    "so3_log_map",
    "so3_rotate_with_step",
    "so3_right_jacobian",
    "so3_right_jacobian_inv",
]

import numpy as np
import math

from ssl_simulator.math import check_and_parse_dimensions, unit_vec

def rot_3d_matrix(roll, pitch, yaw, dec=None):
    """
    Generate R ∈ SO(3) from ROLL, PITCH, YAW.
    Fast for scalar inputs, supports arrays with broadcasting.
    """
    # Convert to arrays but don't force new dimension yet
    roll_arr, pitch_arr, yaw_arr = np.asarray(roll), np.asarray(pitch), np.asarray(yaw)

    # --- SCALAR MODE ---
    if roll_arr.ndim == 0 and pitch_arr.ndim == 0 and yaw_arr.ndim == 0:
        sr, cr = np.sin(roll_arr), np.cos(roll_arr)
        sp, cp = np.sin(pitch_arr), np.cos(pitch_arr)
        sy, cy = np.sin(yaw_arr), np.cos(yaw_arr)

        # Combined rotation matrix (Z-Y-X convention)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr]
        ])
        if dec is not None:
            R = np.round(R, decimals=dec)
        return R

    # --- ARRAY / BROADCAST MODE ---
    roll_arr, pitch_arr, yaw_arr = np.broadcast_arrays(roll_arr, pitch_arr, yaw_arr)

    sr, cr = np.sin(roll_arr), np.cos(roll_arr)
    sp, cp = np.sin(pitch_arr), np.cos(pitch_arr)
    sy, cy = np.sin(yaw_arr), np.cos(yaw_arr)

    shape = roll_arr.shape
    R = np.empty(shape + (3, 3))

    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr

    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr

    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr

    if dec is not None:
        R = np.round(R, decimals=dec)

    return R

def gen_random_rotations(n, seed=None):

    if seed is not None:
        np.random.seed(seed)

    roll  = 2*(np.random.rand((n)) - 0.49) * np.pi # ROLL
    pitch = 2*(np.random.rand((n)) - 0.49) * np.pi # PITCH
    yaw   = 2*(np.random.rand((n)) - 0.49) * np.pi # YAW

    return rot_3d_matrix(roll, pitch, yaw)

def orthonormal_vector_to(v):
    """
    - Select one of the possible perpendicular vector to v ∈ R^3 -
    """
    vx, vy, vz = v[0], v[1], v[2]

    # Capture the singularity
    if (vz < -0.99999999):
        n = np.array([0,-1,0])
    
    # Perpendicular vector computation
    else:
        a = 1/(1 + vz)
        b = -vx*vy*a
        n = np.array([1 - a*vx**2, b, -vx])
    
    return n

def construct_attitude_basis(heading, gravity):
    """
    Construct an orthonormal basis given heading and gravity vectors.
    Handles both single vector (shape (3,) or (1,3)) and batch (shape (N,3)).
    Returns: (3,3) or (N,3,3) basis matrix/matrices.
    """
    heading = check_and_parse_dimensions(heading, (None,3), "heading")
    gravity = check_and_parse_dimensions(gravity, (None,3), "gravity", fill_values=heading.shape[0])

    v1 = unit_vec(heading)
    gravity_proj = gravity - np.sum(gravity * v1, axis=1, keepdims=True) * v1
    v3 = -unit_vec(gravity_proj)
    if v3.any() < 1e-8:
        raise ValueError("Gravity is parallel to heading; cannot construct basis.")
    v2 = -np.cross(v1, v3)
    basis = np.stack((v1, v2, v3), axis=-1)
    return basis

def rotation_matrix_from_vector(v):
    """
    - Given the input vector v, build an orthonormal basis and codify into a rotation matrix R ∈ SO(3) -
    """
    # Normalization of v
    md = v / np.linalg.norm(v)

    # Get an arbitrary (fixed) perperdicular vector
    md_z = -orthonormal_vector_to(md)

    # Compute the las orthogonal vector
    md_y = np.cross(md_z, md)

    # Build the rotation matrix
    R = np.array([md, md_y, md_z]).T
    return R / np.linalg.det(R)

def rotation_angle_from_matrix(R):
    """
    - Compute the distance in the tangent plane (theta) corresponding to a given R ∈ SO(3) -
    Supports single matrix (3,3) or batch (N,3,3).
    """
    # Approximating the exponential map can produce matrices slightly outside SO(3),
    # which may result in |cos(θ)| values greater than 1 due to numerical errors.
    # We address this by applying np.clip to constrain the values within [-1, 1].
    R = np.asarray(R)
    if R.ndim == 2:
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        return theta
    elif R.ndim == 3:
        cos_theta = (np.trace(R, axis1=1, axis2=2) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        return theta
    else:
        raise ValueError("Input must be shape (3,3) or (N,3,3)")

###################################################################
## Isomorphism computation: rotation vector \omega <-> so(3) ######

def so3_hat(omega):
    """
    - Generate \omega_\hat ∈ so(3) from the \omega vector -
    Supports single vector (3,) or batch (N,3).
    """
    omega = np.asarray(omega)
    if omega.ndim == 1:
        wx, wy, wz = omega
        return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    elif omega.ndim == 2:
        wx, wy, wz = omega[:,0], omega[:,1], omega[:,2]
        zeros = np.zeros_like(wx)
        O = np.stack([
            np.stack([zeros, -wz, wy], axis=-1),
            np.stack([wz, zeros, -wx], axis=-1),
            np.stack([-wy, wx, zeros], axis=-1)
        ], axis=1)
        return O
    else:
        raise ValueError("Input must be shape (3,) or (N,3)")

def so3_vee(omega_hat):
    """
    - Generate \omega vector from \omega_\hat ∈ so(3) -
    Supports batch (...,3,3).
    """
    omega_hat = np.asarray(omega_hat)
    wx = omega_hat[...,2,1]
    wy = omega_hat[...,0,2]
    wz = omega_hat[...,1,0]
    return np.stack([wx, wy, wz], axis=-1)

###################################################################

def so3_exp_map(A, n=6, tol=3e-6, warn=True):
    """
    Approximate matrix exponential for so(3) elements using truncated Taylor expansion.

    Args:
        A: skew-symmetric matrix (3x3) or batch (N,3,3).
        n: truncation order (default 6).
        warn: whether to print a warning if rotation step is large.
    """
    A = np.asarray(A)
    
    # Check if A is of shape (3,)
    if A.shape == (3,):
        raise ValueError(
            "Input A has shape (3,). This likely means you are providing an so(3) element in vee form. "
            "Please use so3_hat to convert it to a skew-symmetric matrix before passing it to so3_exp_map."
        )
    
    batch = (A.ndim == 3)
    if not batch:
        A = A[None, ...]  # shape (1,3,3)

    omegas = so3_vee(A)
    thetas = np.linalg.norm(omegas, axis=1)

    # theoretical error estimate
    err_est = thetas**(n+1) / math.factorial(n+1)
    if warn and np.any(err_est > tol):
        import warnings
        warnings.warn(
            f"[so3_exp_map] Some rotation angles θ too large for n={n}. "
            f"Max estimated truncation error={np.max(err_est):.2e} > tol={tol:.1e}. "
            "Consider reducing dt or using Rodrigues' formula."
        )

    N = A.shape[0]
    exp_A = np.tile(np.eye(3), (N,1,1))
    A_i = np.tile(np.eye(3), (N,1,1))
    for i in range(n):
        A_i = np.matmul(A_i, A)
        exp_A = exp_A + A_i / math.factorial(i+1)

    if not batch:
        return exp_A[0]
    return exp_A

def so3_log_map(R, n=5, eps=1e-8): # TODO: test and revise
    """
    Vectorized logarithmic map R ∈ SO(3) -> so(3)
    Handles small and large rotation angles robustly.
    
    Args:
        R: array of shape (..., 3, 3)
        n: number of small-step splits for mid-range rotations
        eps: numerical tolerance to avoid division by zero
    
    Returns:
        log_R: array of shape (..., 3, 3), elements in so(3)
    """
    R = np.asarray(R)
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    N = R_flat.shape[0]
    log_R_flat = np.zeros_like(R_flat)

    # Compute rotation angles
    trace_R = np.trace(R_flat, axis1=1, axis2=2)
    cos_theta = (trace_R - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Masks for different regimes
    small_mask = theta < np.pi/6
    large_mask = theta > 0.998*np.pi
    mid_mask   = ~(small_mask | large_mask)

    # --- SMALL ANGLES ---
    if np.any(small_mask):
        t = theta[small_mask]
        # Use Taylor series: θ / (2 sin θ) ≈ 1/2 + θ^2/12 + 7 θ^4/720
        factor = 0.5 + t**2 / 12 + 7 * t**4 / 720
        RmRt = R_flat[small_mask] - np.transpose(R_flat[small_mask], (0,2,1))
        log_R_flat[small_mask] = factor[:,None,None] * RmRt

    # --- LARGE ANGLES near pi ---
    if np.any(large_mask):
        # Extract axis from diagonal (robust)
        diag = np.stack([R_flat[large_mask,0,0],
                         R_flat[large_mask,1,1],
                         R_flat[large_mask,2,2]], axis=1)
        omega_pi = np.sqrt(np.clip((diag+1)/2, 0, 1))
        # Convert to skew-symmetric
        log_R_flat[large_mask] = so3_hat(omega_pi)

    # --- MID-RANGE ANGLES ---
    if np.any(mid_mask):
        idx = np.where(mid_mask)[0]
        for i in idx:
            Ri = np.eye(3)
            log_R_i = np.zeros((3,3))
            for j in range(n):
                Reval = Ri.T @ R_flat[i]
                trace_eval = np.trace(Reval)
                cos_ti = (trace_eval - 1)/2
                cos_ti = np.clip(cos_ti, -1.0, 1.0)
                theta_i = np.arccos(cos_ti)

                # θ_i / (2 sin θ_i)
                if np.abs(np.sin(theta_i)) < eps:
                    coeff = 1/2  # fallback for tiny sin
                else:
                    coeff = theta_i / (2*np.sin(theta_i))
                # scale for split steps
                coeff /= (n - j)

                log_Ri_Ri = coeff * (Reval - Reval.T)
                log_Ri_I = Ri.T @ log_Ri_Ri @ Ri

                log_R_i += log_Ri_I
                Ri = Ri @ so3_exp_map(log_Ri_I)
            log_R_flat[i] = log_R_i

    # Clean diagonal
    log_R_flat[..., 0,0] = 0
    log_R_flat[..., 1,1] = 0
    log_R_flat[..., 2,2] = 0

    return log_R_flat.reshape(batch_shape + (3,3))

def so3_rotate_with_step(R, omega_hat, step=np.pi/6):
    """
    Vectorized version of rotation update using exponential map in steps.

    Args:
        R: Initial rotation matrix, shape (..., 3, 3)
        omega_hat: Skew-symmetric matrix (so(3)), shape (..., 3, 3)
        step: Maximum rotation angle per step (default: pi/6)

    Returns:
        R_rot: Resulting rotation matrix, same shape as R
    """
    R = np.asarray(R)
    omega_hat = np.asarray(omega_hat)

    # Broadcast R to match omega_hat batch size
    if R.shape[:-2] != omega_hat.shape[:-2]:
        if R.shape[:-2] == (1,):
            # Broadcast R to match omega_hat batch shape
            R = np.broadcast_to(R, omega_hat.shape)
        else:
            raise ValueError(f"Incompatible batch shapes: R {R.shape[:-2]} vs omega_hat {omega_hat.shape[:-2]}")

    # Flatten batch if necessary
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    omega_hat_flat = omega_hat.reshape(-1, 3, 3)
    N = R_flat.shape[0]

    # Compute rotation vectors and angles
    omega_vec = np.stack([omega_hat_flat[:,2,1],
                          omega_hat_flat[:,0,2],
                          omega_hat_flat[:,1,0]], axis=1)
    theta = np.linalg.norm(omega_vec, axis=1)

    # Normalize angles > 2*pi
    mask_large = theta > 2*np.pi
    omega_hat_flat[mask_large] *= (theta[mask_large] % (2*np.pi) / theta[mask_large])[:,None,None]

    # Initialize output
    R_rot_flat = R_flat.copy()

    # Mask for rotations requiring splitting
    mask_split = (theta >= step) & (theta > 1e-8)
    mask_single = ~mask_split

    # --- Single-step rotations ---
    if np.any(mask_single):
        R_rot_flat[mask_single] = np.matmul(
            R_rot_flat[mask_single],
            so3_exp_map(omega_hat_flat[mask_single])
        )

    # --- Multi-step rotations ---
    if np.any(mask_split):
        for i in np.where(mask_split)[0]:
            n_steps = int(theta[i] // step)
            remainder = theta[i] % step
            exp_step = so3_exp_map(step * omega_hat_flat[i] / theta[i])
            R_temp = R_rot_flat[i]
            for _ in range(n_steps):
                R_temp = np.matmul(R_temp, exp_step)
            if remainder > 1e-8:
                R_temp = np.matmul(R_temp, so3_exp_map(remainder * omega_hat_flat[i] / theta[i]))
            R_rot_flat[i] = R_temp

    # Reshape back to original batch shape
    R_rot = R_rot_flat.reshape(batch_shape + (3, 3))
    return R_rot

def so3_right_jacobian(phi):
    """
    Compute the Jacobian of the exponential map at phi ∈ so(3).
    Supports single vector (3,) or batch (N,3).
    """
    phi = np.asarray(phi)
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    I = np.eye(3)

    if phi.ndim == 1:
        phi_hat = so3_hat(phi)
        if theta < 1e-8:
            # Small-angle approximation
            A = 0.5
            B = 1/6
        else:
            A = (1 - np.cos(theta)) / (theta**2)
            B = (theta - np.sin(theta)) / (theta**3)
        J = I - A * phi_hat + B * phi_hat @ phi_hat
        return J
    
    elif phi.ndim == 2:
        J_list = []
        for i in range(phi.shape[0]):
            th = theta[i]
            phi_hat = so3_hat(phi[i])
            if th < 1e-8:
                # Small-angle approximation
                A = 0.5
                B = 1/6
            else:
                A = (1 - np.cos(th)) / (th**2)
                B = (th - np.sin(th)) / (th**3)
            J = I - A * phi_hat + B * phi_hat @ phi_hat
            J_list.append(J)
        return np.array(J_list)
    else:
        raise ValueError("Input must be shape (3,) or (N,3)")

def so3_right_jacobian_inv(phi):
    """
    Compute the Jacobian of the logarithmic map at phi ∈ so(3).
    Supports single vector (3,) or batch (N,3).
    """
    phi = np.asarray(phi)
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    I = np.eye(3)

    if phi.ndim == 1:
        phi_hat = so3_hat(phi)
        if theta < 1e-8:
            # Small-angle approximation
            A = 0.5
            B = 1/12
        else:
            A = 0.5
            B = (1/theta**2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
        J_inv = I + A * phi_hat + B * np.matmul(phi_hat, phi_hat)
        return J_inv
    
    elif phi.ndim == 2:
        J_inv_list = []
        for i in range(phi.shape[0]):
            th = theta[i]
            phi_hat = so3_hat(phi[i])
            if th < 1e-8:
                # Use series expansion for small angles
                A = 0.5
                B = 1/12
            else:
                A = 0.5
                B = (1/th**2) - (1 + np.cos(th)) / (2 * th * np.sin(th))
            J_inv = I + A * phi_hat + B * np.matmul(phi_hat, phi_hat)
            J_inv_list.append(J_inv)
        return np.array(J_inv_list)
    else:
        raise ValueError("Input must be shape (3,) or (N,3)")