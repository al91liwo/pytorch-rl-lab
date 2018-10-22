import numpy as np
from ..common import VelocityFilter
from .base import CartpoleBase, CartPoleDynamics, PhysicSystem


class Cartpole(CartpoleBase):

    def __init__(self, fs, fs_ctrl):
        super(Cartpole, self).__init__(fs, fs_ctrl)
        self._dyn = CartPoleDynamics()
        self._sim_state = None
        self.viewer = None
        self.physics = PhysicSystem(self.timing, x=0, theta=0)

    def _calibrate(self):
        th_in = 0.2* np.random.uniform(-np.pi, np.pi)
        self._state = np.array([0.,th_in,0.,0.])
        self.physics = PhysicSystem(self.timing, x=0, theta=th_in)
        self._sim_state = self.physics.get_state(['x','theta','x_dot','theta_dot'])

    def _sim_step(self, a):
        # Add a bit of noise to action for robustness
        a_noisy = a + 1e-6 * np.float32(
            self._np_random.randn(self.action_space.shape[0]))
        x_ddot, theta_ddot = self._dyn(self._sim_state, a_noisy)
        self.physics.add_acceleration(x=x_ddot, theta=theta_ddot)
        self._sim_state = self.physics.get_state(['x','theta','x_dot','theta_dot'])
        return self._sim_state

    def reset(self):
        self._calibrate()
        return self.step([0.0])[0]


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self._x_lim*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.3365
        cartwidth = 50.0
        cartheight = 30.0

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
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[1] +np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

