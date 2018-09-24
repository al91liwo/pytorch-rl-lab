import numpy as np
from ..common import VelocityFilter
from .base import QubeBase, QubeDynamics


class Qube(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(Qube, self).__init__(fs, fs_ctrl)
        self._sim_state = None
        self._dyn = QubeDynamics()
        self._vp = None
        self._arm = None
        self._pole = None
        self._curve = None

    def _set_gui(self):
        scene_range = 0.2
        arm_radius = 0.003
        arm_length = 0.085
        pole_radius = 0.0045
        pole_length = 0.129
        # http://www.glowscript.org/docs/VPythonDocs/canvas.html
        self._vp.scene.width = 400
        self._vp.scene.height = 300
        self._vp.scene.background = self._vp.color.gray(0.95)
        self._vp.scene.lights = []
        self._vp.distant_light(direction=self._vp.vector(0.2, 0.2, 0.5),
                               color=self._vp.color.white)
        self._vp.scene.up = self._vp.vector(0, 0, 1)
        self._vp.scene.range = scene_range
        self._vp.scene.center = self._vp.vector(0.04, 0, 0)
        self._vp.scene.forward = self._vp.vector(-2, 1.2, -1)
        self._vp.box(pos=self._vp.vector(0, 0, -0.07),
                     length=0.09, width=0.1, height=0.09,
                     color=self._vp.color.gray(0.5))
        self._vp.cylinder(axis=self._vp.vector(0, 0, -1), radius=0.005,
                          length=0.03, color=self._vp.color.gray(0.5))
        # Arm
        arm = self._vp.cylinder()
        arm.radius = arm_radius
        arm.length = arm_length
        arm.color = self._vp.color.blue
        # Pole
        pole = self._vp.cylinder()
        pole.radius = pole_radius
        pole.length = pole_length
        pole.color = self._vp.color.red
        # Curve
        curve = self._vp.curve(color=self._vp.color.white,
                               radius=0.0005,
                               retain=2000)
        return arm, pole, curve

    def _calibrate(self):
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0],
                                        dt=self.timing.dt)
        self._sim_state = 0.02 * np.random.randn(self.state_space.shape[0])
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        thdd, aldd = self._dyn(self._sim_state, a)
        # Update internal simulation state
        self._sim_state[3] += self.timing.dt * aldd
        self._sim_state[2] += self.timing.dt * thdd
        self._sim_state[1] += self.timing.dt * self._sim_state[3]
        self._sim_state[0] += self.timing.dt * self._sim_state[2]
        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[:2]
        vel = self._vel_filt(pos)
        return np.r_[pos, vel]

    def reset(self):
        self._calibrate()
        if self._curve is not None:
            self._curve.clear()
        return self.step(0.0)[0]

    def render(self, mode='human'):
        if self._vp is None:
            import importlib
            self._vp = importlib.import_module('vpython')
            self._arm, self._pole, self._curve = self._set_gui()
        th, al, _, _ = self._state
        arm_pos = (self._dyn.Lr * np.cos(th),
                   self._dyn.Lr * np.sin(th),
                   0.0)
        pole_ax = (-self._dyn.Lp * np.sin(al) * np.sin(th),
                   self._dyn.Lp * np.sin(al) * np.cos(th),
                   -self._dyn.Lp * np.cos(al))
        self._arm.axis = self._vp.vector(*arm_pos)
        self._pole.pos = self._vp.vector(*arm_pos)
        self._pole.axis = self._vp.vector(*pole_ax)
        self._curve.append(self._pole.pos + self._pole.axis)
        self._vp.rate(self.timing.render_rate)
