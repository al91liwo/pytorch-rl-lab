import numpy as np
import vpython as vp
from qube.base import QubeBase


class Qube(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(Qube, self).__init__(fs, fs_ctrl)
        self._arm, self._pole, self._curve = self._set_gui()

    @staticmethod
    def _set_gui():
        scene_range = 0.2  # 0.4m x 0.4m excerpt of the scene
        arm_radius = 0.003
        arm_length = 0.085
        pole_radius = 0.0045
        pole_length = 0.129

        # Vpython scene: http://www.glowscript.org/docs/VPythonDocs/canvas.html
        vp.scene.width = 800
        vp.scene.height = 800
        vp.scene.background = vp.color.gray(0.95)  # higher -> brighter
        vp.scene.lights = []
        vp.distant_light(direction=vp.vector(0.2, 0.2, 0.5),
                         color=vp.color.white)
        vp.scene.up = vp.vector(0, 0, 1)  # z-axis is showing up
        vp.scene.range = scene_range
        vp.scene.center = vp.vector(0.04, 0, 0)
        vp.scene.forward = vp.vector(-2, 1.2, -1)

        vp.box(pos=vp.vector(0, 0, -0.07), length=0.09, width=0.1, height=0.09,
               color=vp.color.gray(0.5))
        vp.cylinder(axis=vp.vector(0, 0, -1), radius=0.005, length=0.03,
                    color=vp.color.gray(0.5))

        # Arm
        arm = vp.cylinder()
        arm.radius = arm_radius
        arm.length = arm_length
        arm.color = vp.color.blue

        # Pole
        pole = vp.cylinder()
        pole.radius = pole_radius
        pole.length = pole_length
        pole.color = vp.color.red

        # Curve
        curve = vp.curve(color=vp.color.white, radius=0.0005, retain=2000)
        return arm, pole, curve

    def _fw_kin(self, th, al):
        reachable = False
        # check if theta is in reachable space
        if np.abs(th) <= self.state_max[0]:
            x = - self.Lp * np.sin(al) * np.sin(th) + self.Lr * np.cos(th)
            y = self.Lp * np.sin(al) * np.cos(th) + self.Lr * np.sin(th)
            z = - self.Lp * np.cos(al)
            reachable = True
        # if theta is not reachable, return some default values
        else:
            x, y, z = [0, 0, 0]
        return [x, y, z], reachable

    def _sim_step(self, x, a):
        th, al, th_d, al_d = x

        tau = self.km * (a - self.km * th_d) / self.Rm

        c1 = self.a1 + self.a2 * (1.0 - np.cos(2 * al)) / 4 + self.Jr
        c2 = self.a3 * np.cos(al)
        c3 = tau - self.Dr * th_d \
             - self.a2 * np.sin(2 * al) / 2 * th_d * al_d \
             + self.a3 * np.sin(al) * al_d ** 2
        c4 = - self.Dp * al_d + self.a2 * np.sin(2 * al) * th_d ** 2 / 4 \
             - self.a5 * np.sin(al)
        c5 = self.a4 * c1 - c2 * c2

        th_dd = (c3 * self.a4 - c2 * c4) / c5
        al_dd = (c1 * c4 - c2 * c3) / c5

        th_d += self._dt * th_dd
        al_d += self._dt * al_dd
        th += self._dt * th_d
        al += self._dt * al_d

        return np.r_[th, al, th_d, al_d]

    def reset(self):
        self._state = 0.1 * np.random.randn(self.state_space.shape[0])
        self._curve.clear()
        return self.step(0.0)[0]

    def render(self, mode='human'):
        # Position: End of the arm
        x_pos_arm = self.Lr * np.cos(self._state[0])
        y_pos_arm = self.Lr * np.sin(self._state[0])
        z_pos_arm = 0

        pos_axis, reachable = self._fw_kin(self._state[0], self._state[1])
        if not reachable:
            return

        # End of the arm -to- End of the pole (normalization not needed)
        x_axis_pole = pos_axis[0] - x_pos_arm
        y_axis_pole = pos_axis[1] - y_pos_arm
        z_axis_pole = pos_axis[2] - z_pos_arm

        # render the computed positions
        self._arm.axis = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self._pole.pos = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self._pole.axis = vp.vector(x_axis_pole, y_axis_pole, z_axis_pole)

        self._curve.append(self._pole.pos + self._pole.axis)

        vp.rate(int(self._fs_ctrl))
