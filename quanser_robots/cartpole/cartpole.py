import numpy as np
from ..common import VelocityFilter, PhysicSystem, Simulation, Timing, NoFilter
from .base import CartpoleBase

class Cartpole(Simulation, CartpoleBase):

    def __init__(self, fs, fs_ctrl, long_pole=False, **kwargs):
        wcf = 62.8318
        zetaf = 0.9
        CartpoleBase.__init__(self, fs, fs_ctrl, **kwargs)
        
        if self.stabilization:
            theta_init = lambda: np.random.choice([np.random.uniform(-np.pi, np.pi+0.1),
                                                   np.random.uniform(np.pi -0.1, np.pi)])
        else:
            theta_init = lambda: 0.01 * np.random.uniform(-np.pi, np.pi)

        Simulation.__init__(self, fs,
                                      fs_ctrl,
                                      dynamics=CartPoleDynamics(long=long_pole),
                                      entities=['x', 'theta'],
                                      filters={
                                          'x':NoFilter(dt=self.timing.dt),
                                          'theta':NoFilter(dt=self.timing.dt)
                                      },
                                      initial_distr={
                                          'x': lambda: 0.,
                                          'theta': theta_init
                                      })


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self._dynamics._x_lim*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = scale *0.04 * self._dynamics._pl
        polelen = scale * self._dynamics._pl
        cartwidth = scale *0.3 * self._dynamics._pl
        cartheight = scale *0.2 * self._dynamics._pl

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self._sim_state is None: return None

        x = self._sim_state
        cartx = x[0]*scale+screen_width/2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[1]-np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class CartPoleDynamics:

    def __init__(self, long=False):

        self._eta_m = 1.            # Motor efficiency  []
        self._eta_g = 1.             # Planetary Gearbox Efficiency []
        self._Kg = 3.71             # Planetary Gearbox Gear Ratio
        self._Jm = 3.9E-7           # Rotor inertia [kg.m^2]
        self._r_mp = 1.30E-3        # Motor Pinion radius [m] #TODO: was 6.35E-3
        self._Rm = 2.6              # Motor armature Resistance [Ohm]
        self._Kt = .00767           # Motor Torque Constant [N.zz/A]
        self._Km = .00767           # Motor Torque Constant [N.zz/A]

        self._mc = 0.38  # mass of the cart [kg]
        if long:
            self._mp = 0.23         # mass of the pole [kg]
            self._pl = 0.641 / 2.   # half of the pole lenght [m]
        else:
            self._mp = 0.127  # mass of the pole [kg]
            self._pl = 0.3365 / 2.  # half of the pole lenght [m]

        self._Jp = self._pl**2 * self._mp/3.   # Pole inertia [kg.m^2]
        self._Jeq = self._mc + (self._eta_g * self._Kg**2 * self._Jm)/(self._r_mp**2)
        self._JT = self._Jeq * self._Jp + self._mp * self._Jp + self._Jeq * self._mp * self._pl**2

        self._Beq = 5.4             # Equivalent Viscous damping Coefficient
        self._Bp = 0.0024           # Viscous coefficient at the pole

        self._x_lim = 0.814 / 2.  # limit of position of the cart [m]

        self._g = 9.81  # gravitational acceleration [m.s^-2]

    def __call__(self, s, V_m):
        x, theta, x_dot, theta_dot = s

        F = (self._eta_g * self._Kg * self._eta_m * self._Kt) / (self._Rm * self._r_mp) * (
                   -  self._Kg * self._Km * x_dot / self._r_mp + self._eta_m * (V_m)) #- self._Beq*x_dot

        A = np.array([[np.cos(theta), self._pl],
                      [self._mp + self._mc, self._pl * self._mp * np.cos(theta)]])

        b = np.array([-self._g * np.sin(theta), # TODO: from - self. bla bla it is new
                      F + self._mp * self._pl * theta_dot ** 2 * np.sin(theta)])# - 0.1 * theta_dot])

        x_ddot, alpha_ddot = np.linalg.solve(A, b)

        return np.array([x_ddot, alpha_ddot])

