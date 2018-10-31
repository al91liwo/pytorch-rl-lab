import numpy as np
from ..common import VelocityFilter, PhysicSystem, Simulation, Timing, NoFilter
from .base import DoublePendulumBase

class DoublePendulum(Simulation, DoublePendulumBase):

    def __init__(self, fs, fs_ctrl, long_pole=False):
        zetaf = 0.9

        wcf_1 = 2*np.pi*50.
        wcf_2 = 2*np.pi*10.

        DoublePendulumBase.__init__(self, fs, fs_ctrl)
        Simulation.__init__(self, fs,
                                      fs_ctrl,
                                      dynamics=DoublePendulumDynamics(long=long_pole),
                                      entities=['x', 'theta1', 'theta2'],
                                      # filters={
                                      #     'x':VelocityFilter(2, num=(wcf_1**2, 0), den=(1, 2*wcf_1*zetaf, wcf_1**2),
                                      #                                                        x_init=np.array([0.]), dt=self.timing.dt),
                                      #     'theta1':VelocityFilter(2, num=(wcf_2**2, 0), den=(1, 2*wcf_2*zetaf, wcf_2**2),
                                      #                            x_init=np.array([0.]), dt=self.timing.dt),
                                      #     'theta2':VelocityFilter(2, num=(wcf_2**2, 0), den=(1, 2*wcf_2*zetaf, wcf_2**2),
                                      #                            x_init=np.array([0.]), dt=self.timing.dt)
                                      # },
                                        filters={
                                            'x': NoFilter(dt=self.timing.dt),
                                            'theta1': NoFilter(dt=self.timing.dt),
                                            'theta2': NoFilter(dt=self.timing.dt)
                                        },
                                      initial_distr={
                                          'x': lambda: 0.,
                                          'theta1': lambda: 0.001 * np.random.uniform(-1.,1.),
                                          'theta2': lambda: 0.001 * np.random.uniform(-1.,1.)
                                      })


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self._dynamics._x_lim*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = scale *0.04 * self._dynamics._pl
        polelen1 = scale * self._dynamics.l1
        polelen2 = scale * self._dynamics.l2
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

            l,r,t,b = -polewidth/2,polewidth/2,polelen1-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))

            l,r,t,b = -polewidth/2,polewidth/2,polelen2-polewidth/2,-polewidth/2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pole2trans = rendering.Transform(translation=(0, axleoffset))
            pole2.set_color(.8, .6, .4)

            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.carttrans)

            self.viewer.add_geom(pole)
            self.viewer.add_geom(pole2)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            #self.axle.add_attr(self.pole2trans)
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
        self.poletrans.set_rotation(x[1])
        self.pole2trans.set_rotation(x[2]+x[1])
        self.pole2trans.set_translation(-polelen1*np.sin(x[1]), polelen1*np.cos(x[1]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class DoublePendulumDynamics:

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

        self._Beq = 4.3            # Equivalent Viscous damping Coefficient
        self._Bp = 0.0024           # Viscous coefficient at the pole

        self.m1, self.m2 = 0.072, 0.127
        self.l1, self.l2 = 0.1143, 0.1778
        self.Bp1 = self.Bp2 = 0.0024


        self._x_lim = 0.814 / 2.  # limit of position of the cart [m]

        self._g = 9.81  # gravitational acceleration [m.s^-2]

    def __call__(self, s, V_m):
        x, alpha1, alpha2,  x_dot, alpha1_dot, alpha2_dot = s

        # Transformation to the system used in the dynamics
        theta1 = -alpha1
        theta2 = -alpha2-alpha1
        theta1_dot = -alpha1_dot
        theta2_dot = -alpha2_dot - alpha1_dot

        F = (self._eta_g * self._Kg * self._eta_m * self._Kt) / (self._Rm * self._r_mp) * (
                   -self._Kg * self._Km * x_dot / self._r_mp + self._eta_m * V_m)

        m, m1, m2 = self._mc, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        Bp1, Bp2 = self.Bp1, self.Bp2
        Beq = self._Beq

        A = np.array([[m+m1+m2, l1*(m1+m2)*np.cos(theta1), m2*l2*np.cos(theta2)],
                      [l1*(m1+m2)*np.cos(theta1), l1**2*(m1+m2), l1*l2*m2*np.cos(theta1-theta2)],
                      [l2*m2*np.cos(theta2), l1*l2*m2*np.cos(theta1-theta2), l2**2*m2]], dtype=np.float64)

        b = np.array([l1*(m1+m2)*theta1_dot**2*np.sin(theta1) + m2*l2*theta2_dot**2*np.sin(theta2) + F- Beq*x_dot,
                      -l1*l2*m2*theta2_dot**2*np.sin(theta1-theta2) + self._g*(m1+m2)*l1*np.sin(theta1)- Bp1*theta1_dot,
                      l1*l2*m2*theta1_dot**2*np.sin(theta1-theta2) + self._g*l2*m2*np.sin(theta2) - Bp2*(theta2_dot)
                      ], dtype=np.float64)

        x_ddot, theta1_ddot, theta2_ddot = np.linalg.solve(A, b)

        # Transformation to the system used externally
        alpha1_ddot = -theta1_ddot
        alpha2_ddot = -theta2_ddot - alpha1_ddot

        return np.array([x_ddot, alpha1_ddot, alpha2_ddot])

