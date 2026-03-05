# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from gymnasium.envs.classic_control import MountainCarEnv
import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class FancyMountainCar(MountainCarEnv):  # piggybag on the original env.
    visualization = None

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)

    def render(self):
        if self.visualization is None:
            self.visualization = MountainCarVisualization(self, self.agent if hasattr(self, 'agent') else None)
        return self.visualization.render()

    def close(self):
        if self.visualization is not None:
            self.visualization.close()


from irlc.pacman.pacman_resources import WHITE, BLACK
from irlc.utils.graphics_util_pygame import GraphicsUtilGym
class MountainCarVisualization:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        # self.k = 0
        # self.states = []
        # self.actions = []
        # self.factories = []
        # self.inventory = inventory
        # xmin = -0.2
        # xmax = inventory.N * 2 + 1.4
        # xmax = 4

        # ymin = -0.4
        # ymax = 1.4 + 0.2
        # dx = xmax - xmin
        # dy = ymax - ymin
        self.ga = GraphicsUtilGym()
        # screen_width = 1300
        screen_width = env.screen_width * 2
        #
        # -env.min_position
        # env.max_position

        xmin = env.min_position
        xmax = env.max_position + 1.8
        # env._height

        screen_height = env.screen_height
        ymin = 0
        ymax = 1.2
        # screen_height = dy * (screen_width / dx)
        frames_per_second = 30
        self.ga.begin_graphics(screen_width, screen_height,
                               local_xmin_xmax_ymin_ymax=(xmin, xmax, ymax, ymin), frames_per_second=frames_per_second,
                               color=WHITE, title=f"MountainCar Environment")

        # self.last_action = None
        # self.agent = None
        # self.last_reward = None
        # self.scale = screen_width / dx

    x_cache = []


    def render(self):
        # if self.env.render_mode is None:
        #     assert self.env.spec is not None
        #     gym.logger.warn(
        #         "You are calling render method without specifying any render mode. "
        #         "You can specify the render_mode at initialization, "
        #         f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
        #     )
        #     return
        # try:
        #     import pygame
        #     from pygame import gfxdraw
        # except ImportError as e:
        #     raise DependencyNotInstalled(
        #         'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
        #     ) from e

        #
        #
        # if self.screen is None:
        #     pygame.init()
        #     if self.render_mode == "human":
        #         pygame.display.init()
        #         self.screen = pygame.display.set_mode(
        #             (self.screen_width, self.screen_height)
        #         )
        #     else:  # mode in "rgb_array"
        #         self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # if self.clock is None:
        #     self.clock = pygame.time.Clock()
        self.ga.draw_background()
        # self.ga.circle("sadf", pos=(0,0), r=100, fillColor=(100, 10, 50))

        pos = self.env.state[0]
        scale = 1

        xs = np.linspace(self.env.min_position, self.env.max_position, 100)
        ys = self.env._height(xs)
        # xys = list(zip((xs - self.env.min_position) * scale, ys * scale))

        self.ga.polyline("asdfasfd", xs=xs, ys=ys, width=1)


        # pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))



        world_width = self.env.max_position - self.env.min_position
        # scale = self.screen_width / world_width
        rscale = self.env.screen_width / world_width

        carwidth = 40 / rscale
        carheight = 20 / rscale

        # self.surf = pygame.Surface((self.screen_width, self.screen_height))
        # self.surf.fill((255, 255, 255))

        # pos = self.state[0]

        # xs = np.linspace(self.min_position, self.max_position, 100)
        # ys = self._height(xs)
        # xys = list(zip((xs - self.min_position) * scale, ys * scale))

        # pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))
        import pygame
        clearance = 10 / rscale
        # clearance=0.01

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - 0*self.env.min_position) * scale,
                    c[1] + clearance + self.env._height(pos) * scale,
                )
            )
        self.ga.polygon("adsfasdf", coords=coords, outlineColor=BLACK, fillColor=BLACK, width=2)
        # gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        # gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))


        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                c[0] + (pos - 0*self.env.min_position) * scale,
                c[1] + clearance + self.env._height(pos) * scale,
            )

            # gfxdraw.aacircle(
            #     self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            # )

            self.ga.circle("asdf", (wheel[0], wheel[1]),  int(carheight / 2.5*rscale), fillColor=(128, 128, 128), outlineColor= (70, 70, 70))
            #
            # gfxdraw.filled_circle(
            #     self.surf, wheel[0], wheel[1], int(carheight / 2.5 * rscale), (128, 128, 128)
            # )

        flagx = (self.env.goal_position - 0*self.env.min_position) * scale
        flagy1 = self.env._height(self.env.goal_position) * scale
        flagy2 = flagy1 + 50/rscale
        self.ga.line("asdfasdf", (flagx, flagy1), (flagx, flagy2), color=(0, 0, 0))

        self.ga.polygon(
                "sdfasdf",
            [(flagx, flagy2), (flagx, flagy2 - 10/rscale), (flagx + 25/rscale, flagy2 - 5/rscale)],
            (204, 204, 0),
        )
        # gfxdraw.aapolygon(
        #     self.surf,
        #     [(flagx, flagy2), (flagx, flagy2 - 10/rscale), (flagx + 25/rscale, flagy2 - 5/rscale)],
        #     (204, 204, 0),
        # )
        # gfxdraw.filled_polygon(
        #     self.surf,
        #     [(flagx, flagy2), (flagx, flagy2 - 10/rscale), (flagx + 25/rscale, flagy2 - 5)],
        #     (204, 204, 0),
        # )
        # Optionally draw the value functino.
        # oxmin = 0.6
        # oxmax = 1.7
        # oymin = 0
        # oymax = 1

        # self.env.observation_space
        # dx = 1.5
        # dy = 0

        # sX = 1
        # sY = 1

        # Pscale = 1
        Vscale = 6

        # def pos2s(pos):#, vel):
        #     return pos + 1.8 #, (vel + 0.2) * 3
        # def vel2s(vel):
        #     return (vel + 0.) * Vscale

        def x2s(pos, vel):
            return pos + 1.75, (vel + 0.1) * Vscale

        xmin,ymin = x2s(self.env.observation_space.low[0], self.env.observation_space.low[1] )
        xmax,ymax = x2s(self.env.observation_space.high[0], self.env.observation_space.high[1] )

        px, py = x2s( *np.asarray(self.env.state).tolist())



        # self.env.observation_space.low
        if self.agent is not None:

            def colfunc(val, minval, maxval, startcolor, stopcolor):
                """ Convert value in the range minval...maxval to a color in the range
                    startcolor to stopcolor. The colors passed and the one returned are
                    composed of a sequence of N component values (e.g. RGB).
                """
                f = float(val - minval) / (maxval - minval)
                return tuple( float( f * (b - a) + a) for (a, b) in zip(startcolor, stopcolor))

            RED, YELLOW, GREEN = (1, 0, 0), (1, 1, 0), (0, 1, 0)
            CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
            steps = 10
            minval, maxval = 0.0, 1.0
            # incr = (maxval - minval) / steps
            # for i in range(steps + 1):
            #     val = minval + round(i * incr, 1)
            #     # print('{:.1f} -> ({:.3f}, {:.3f}, {:.3f})'.format(
            #     #     val, *colfunc(val, minval, maxval, BLUE, RED)))

            value_function = lambda s: -max(self.agent.Q.get_Qs(s)[1])

            grid_size = 40
            # grid_size = 30
            low = self.env.unwrapped.observation_space.low
            high = self.env.unwrapped.observation_space.high
            X, Y = np.meshgrid(np.linspace(low[0], high[0], grid_size), np.linspace(low[1], high[1], grid_size))
            Z = X * 0

            if self.x_cache is None or len(self.x_cache) == 0:
                for i, (x, y) in enumerate(zip(X.flat, Y.flat)):
                    s = (x, y)
                    xx = [self.agent.Q.x(s, a) for a in range(self.env.action_space.n) ]
                    self.x_cache.append(xx)
                    # Z.flat[i] = value_function((x, y))
                pass
            # for i, (x, y) in enumerate(zip(X.flat, Y.flat)):
            #     # [max([float(self.agent.Q.w @ dx) for dx in xx]) for xx in self.x_cache]
            #
            #
            #
            #     Z.flat[i] = value_function((x, y))
            # pass
            for i in range(len(self.x_cache)):
                Z.flat[i] = max([float(self.agent.Q.w @ dx) for dx in self.x_cache[i]])
            pass

            for i in range(len(Z.flat)):
                ddx = (X.max() - X.min()) / (grid_size-1)
                ddy = (Y.max() - Y.min()) / (grid_size-1)

                z = colfunc(Z.flat[i], Z.min(), Z.max()+0.01, BLUE, RED)

                z = tuple( int(x*255) for x in z)

                xmin, ymin = x2s(X.flat[i], Y.flat[i])
                xmax, ymax = x2s(X.flat[i]+ddx, Y.flat[i]+ddy)

                self.ga.rectangle(color=z, x=xmin, y=ymin, width=xmax-xmin, height=ymax-ymin)
            pass
            # colfunc(val, minval, maxval, startcolor, stopcolor):

        self.ga.rectangle(color=BLACK, x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin, border=1)
        self.ga.circle("asdf", (px, py), r=5, fillColor=(200, 200, 200))

        return self.ga.blit(render_mode=self.env.render_mode)

        # self.surf = pygame.transform.flip(self.surf, False, True)
        # self.screen.blit(self.surf, (0, 0))
        # if self.render_mode == "human":
        #     pygame.event.pump()
        #     self.clock.tick(self.metadata["render_fps"])
        #     pygame.display.flip()
        #
        # elif self.render_mode == "rgb_array":
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     )

    def close(self):
        self.ga.close()

if __name__ == '__main__':
    from irlc import Agent, interactive, train
    env = FancyMountainCar(render_mode='human')
    num_of_tilings = 8
    alpha = 0.3
    from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
    # env = gym.make("MountainCar-v0")
    agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha/num_of_tilings, epsilon=0)
    # agent = Agent(env)

    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=10)

    env.close()



    pass
