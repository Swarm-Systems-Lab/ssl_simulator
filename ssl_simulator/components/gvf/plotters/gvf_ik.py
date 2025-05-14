"""
"""

import numpy as np
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import PlotterVF

#######################################################################################

class PlotterGvfIk(PlotterVF):
    def __init__(self, gvf_traj, ax, **kwargs):
        self.gvf_traj = gvf_traj
        self.ax = ax

        self.xy_offset = None
        self.mapgrad_pos = None
        self.mapgrad_vec = None
        self.cond_xx, self.cond_yy = None, None
        self.cond_grid = None

        self.speed = None
        self.gamma = None
        self.gamma_dot = None

        self.quivers = None
        self.cond_map = None

    def draw(self, draw_field=True, s=1, ke=1.0, pts=30, pts_cond=200, xy_offset=[0,0], 
             gamma=0, gamma_dot=0, speed=1, **kwargs):
        """
        Plot the trajectory and the vector field.
        """
        self.s = s
        self.ke = ke
        self.xy_offset = xy_offset
        self.gamma = gamma
        self.gamma_dot = gamma_dot
        self.speed = speed

        # Config
        self.kw_line = parse_kwargs(kwargs, dict(ls="--", lw=1, c="k"))
        self.kw_vect.update(dict(lw=1, zorder=4, alpha=0.2, color="black"))
        self.kw_vect = parse_kwargs(kwargs, self.kw_vect)

        # Plot trajectory
        traj = self.gvf_traj.gen_param_points()
        self.ax.plot(*xy_offset, "+k", zorder=0)
        traj_plot, = self.ax.plot(traj[0], traj[1], zorder=0, **self.kw_line)

        # Plot vector field
        if draw_field:
            self.mapgrad_pos = self._compute_xymesh(pts, **kwargs)
            self.cond_xx, self.cond_yy = self._compute_xymesh(pts_cond, return_mesh=True, **kwargs)
            self.cond_grid = np.column_stack([self.cond_xx.flatten(), self.cond_yy.flatten()])

            self.cond_flags = self._compute_cond_flag_map()
            self.mapgrad_vec = self._compute_vector_field()

            self.quivers = self._draw_field(self.ax)
            self.cond_map = self.ax.pcolormesh(self.cond_xx, self.cond_yy, self.cond_flags, 
                                               shading='auto', cmap='Greys', alpha=0.2, zorder=0)

        return traj_plot
    
    def update(self, gamma, gamma_dot, speed):
        self.gamma = gamma
        self.gamma_dot = gamma_dot
        self.speed = speed

        if self.mapgrad_pos is not None:
            self.mapgrad_vec = self._compute_vector_field()
            self.quivers.set_UVC(self.mapgrad_vec[:,0], self.mapgrad_vec[:,1])

        if self.cond_xx is not None:
            self.cond_flags = self._compute_cond_flag_map()
            # self.cond_flags = self.cond_flags + 1
            self.cond_map.set_array(self.cond_flags)

    # ---------------------------------------------------------------------------------

    def _compute_vector_field(self):
        """
        Generate the vector field to be plotted.
        """
        pos = self.mapgrad_pos

        phi_vals  = self.gvf_traj.phi_vec(pos)
        grads  = self.gvf_traj.grad_vec(pos)

        J1 = grads[:, 0]
        J2 = grads[:, 1]
        J_Jt = J1**2 + J2**2

        # --- Vectorized "check_alpha" condition ---
        u = -self.ke * (phi_vals + self.gamma)
        un_x_cond = J1 / J_Jt * (u - self.gamma_dot)
        un_y_cond = J2 / J_Jt * (u - self.gamma_dot)
        un_norm2 = un_x_cond**2 + un_y_cond**2

        cond_flags = un_norm2 < self.speed**2

        # --- Compute error and control input based on condition ---
        e = np.where(cond_flags, phi_vals + self.gamma, phi_vals)
        e_tdot = np.where(cond_flags, self.gamma_dot, 0.0)
        uc_ub = -self.ke * e - e_tdot

         # --- Normal component (recomputed using the control term) ---
        un_x = J1 / J_Jt * uc_ub
        un_y = J2 / J_Jt * uc_ub
        un_norm2 = un_x**2 + un_y**2
        un_norm = np.sqrt(un_norm2)

         # --- Tangential component based on the scalar field gradients ---
        ut_x = self.s * J2
        ut_y = -self.s * J1
        ut_norm = np.sqrt(ut_x**2 + ut_y**2)
        ut_hat_x = ut_x / ut_norm
        ut_hat_y = ut_y / ut_norm

        # --- Combine normal and tangential terms ---
        alpha = np.sqrt(np.clip(self.speed**2 - un_norm2, a_min=0, a_max=None))
        pd_dot_x = np.where(cond_flags, alpha * ut_hat_x + un_x,
                                    self.speed * un_x / un_norm)
        pd_dot_y = np.where(cond_flags, alpha * ut_hat_y + un_y,
                                    self.speed * un_y / un_norm)
        pd = np.stack((pd_dot_x, pd_dot_y), axis=1)

        # Normalize for plotting
        norm = np.linalg.norm(pd, axis=1, keepdims=True)
        vec_normalized = pd / np.clip(norm, 1e-8, None)
        return vec_normalized
    
    def _compute_cond_flag_map(self):
        pos = self.cond_grid

        phi_vals  = self.gvf_traj.phi_vec(pos)
        grads  = self.gvf_traj.grad_vec(pos)

        J1 = grads[:, 0]
        J2 = grads[:, 1]
        J_Jt = J1**2 + J2**2

        # --- Vectorized "check_alpha" condition ---
        u = -self.ke * (phi_vals + self.gamma)
        un_x_cond = J1 / J_Jt * (u - self.gamma_dot)
        un_y_cond = J2 / J_Jt * (u - self.gamma_dot)
        un_norm2 = un_x_cond**2 + un_y_cond**2

        cond_flags = un_norm2 < self.speed**2
        cond_flags_2d = cond_flags.reshape(self.cond_xx.shape).astype(float)
        return cond_flags_2d
    
#######################################################################################