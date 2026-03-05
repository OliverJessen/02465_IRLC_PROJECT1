# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import pygame
from irlc.ex01.inventory_environment import InventoryEnvironment
from irlc.utils.graphics_util_pygame import formatColor

class VizInventoryEnvironment(InventoryEnvironment):
    """This is a variant of the Inventory environment which also sets up visualization.
    Most of the additional code involves calling the visualization and setting up keybindings. The one small change is that Gymnasium
    typically reset immediately on reaching the final state. I.e., the terminal state will typically not be rendered. """

    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': 30
                }

    def __init__(self, N=2, render_mode='human'):
        super(VizInventoryEnvironment, self).__init__(N)
        self.render_mode = render_mode
        self.viewer = None
        self.in_term_state = False

    def get_keys_to_action(self):
        k2a = [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3]
        k2a = {(k2a[i],) : i for i in range(self.action_space.n) }
        return k2a

    def reset(self):
        s, info = super().reset()
        self.s = s
        self.action = None
        self.w = None
        self.reward = None
        self.render()
        return s, info

    def step(self, a):
        self.action = a
        print(f"Step using {a=}")
        if self.in_term_state:
            self.reward = 0
            self.k += 1
            self.in_term_state = False
            return self.s, 0, True, False, {}
        else:
            # s_next, reward, terminated, trunctated, info = super().step(a)
            w = np.random.choice(3, p=(.1, .7, .2))  # Generate random disturbance
            self.w = w
            s_next = max(0, min(2, self.s - w + a))
            reward = -(a + (self.s + a - w) ** 2)  # reward = -cost      = -g_k(x_k, u_k, w_k)
            terminated = self.k == self.N - 1  # Have we terminated? (i.e. is k==N-1)
            self.s = s_next  # update environment state
            self.k += 1
            if terminated:
                self.in_term_state = True
            self.reward = reward
            return s_next, reward, False, False, {}


    def render(self, mode='human', agent=None, prev_action=None, reward=None):
        if self.viewer is None:
            self.viewer = InventoryViewer(self, frames_per_second=self.metadata['render_fps'])
        print(f"render: {self.action=}")
        self.viewer.update(self.agent, state=self.s, k=self.k, action=self.action, reward=self.reward, w=self.w, restart=self.action is None)
        return self.viewer.blit(render_mode=self.render_mode) #(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.viewer.close()


from irlc.pacman.pacman_resources import WHITE, BLACK, Ghost
from irlc.utils.graphics_util_pygame import GraphicsUtilGym

class InventoryViewer:
    scale = 400  # Scale of a single bar.
    width = 0.4 * scale  # with of a bar.

    def __init__(self, inventory : InventoryEnvironment, frames_per_second=None):
        # print("BEGINNING GRAPHICS")
        self.k = 0
        self.states = []
        self.actions = []
        self.factories = []
        self.inventory = inventory
        xmin = -0.2
        xmax = inventory.N*2 + 1.4
        ymin = -0.4
        ymax = 1.4
        dx = xmax-xmin
        dy = ymax-ymin
        self.ga = GraphicsUtilGym()
        screen_width = 1300
        self.ga.begin_graphics(screen_width, dy * (screen_width / dx), local_xmin_xmax_ymin_ymax=(xmin, xmax, ymax, ymin), frames_per_second=frames_per_second, color=formatColor(0, 0, 0), title=f"Inventory environment using N={inventory.N}")
        self.last_action = None
        self.agent = None
        self.last_reward = None
        self.scale = screen_width / dx


    def close(self):
        self.ga.close()

    def blit(self, render_mode='human'):
        return self.ga.blit(render_mode=render_mode)

    def master_render(self):
        self.ga.draw_background()
        for i, factory in enumerate(self.factories):
            factory.render()

        if hasattr(self.inventory, '_interactive_data') and 'avg_reward_per_episode' in self.inventory._interactive_data:
            avg_reward = self.inventory._interactive_data['avg_reward_per_episode']
            episodes = self.inventory._interactive_data['completed_episodes']
            self.ga.text("sadf", (0.1, -0.1), WHITE, contents=f"Completed episodes = {episodes}",
                         size=12,
                         style='bold', anchor='w')
            self.ga.text("sadf", (0.1, -0.2), WHITE, contents=f"Average reward per episode = {avg_reward:.2f}",
                         size=12,
                         style='bold', anchor='w')



    def update(self, agent, k, state, action, reward, w, restart=False):
        self.agent = agent
        if restart:
            # print("Restarting the sim now..")
            self.factories = [Factory(graphics_adaptor=self.ga, x=0, y=0, k=0, state=state)]

        if len(self.factories) <= k:
            self.factories.append(Factory(graphics_adaptor=self.ga, x=k*2, y=0, k=k, state=state))

            if len(self.factories) <= self.inventory.N+1:
                # print("Setting actions.")
                self.factories[k-1].action = action
                self.factories[k-1].w = w
                self.factories[k-1].reward = reward


        self.master_render()


class Factory:
    def __init__(self, graphics_adaptor, x, y, order=1, scale=10., k=1, state=2):
        self.ga = graphics_adaptor
        self.x = x
        self.y = y

        self.scale = scale
        self.s = state
        self.action = None
        self.reward = None
        self.w = None
        self.k = k

    def render(self):
        self.ga.rectangle(color=WHITE, x=self.x, y=0, width=1, height=1, border=1)
        self.ga.text("sadf", (self.x + 0.5, 1.1), WHITE, contents=f"day k = {self.k}",
                     size=12,
                     style='bold', anchor='c')

        self.ga.text("sadf", (self.x + 0.5, 0.8), WHITE, contents=f"state s_{self.k} = {self.s}",
                     size=12,
                     style='bold', anchor='c')

        mw = 1
        dh = 0.1

        rad = mw / 3 / 2
        for h in range(self.s):
            loc = self.x + rad * 2 * h + rad + mw * (3 - self.s) / 3 / 2
            self.ga.circle("sadf", (loc, rad), 0.8 * rad * self.ga.scale(), fillColor=WHITE)

        scale = self.ga.scale()

        if self.action is not None:
            self.ga.text("sdaf", (self.x + 1.5, 0.8 + dh), WHITE, contents=f"action = {self.action}", size=12, style="bold", anchor="c")
            # self.ga.line("sadf", (self.x+1.1, 0.5 + dh), (self.x+1.8, 0.5+dh), color=WHITE, width=2)
            self.ga.line("sadf", (self.x + 1.1, 0.5 + dh), (ex := self.x + 1.9, ey := 0.5 + dh), color=WHITE, width=2)

            self.ga.line("sadf", (ex, ey), (ex-0.05, ey-0.05), color=WHITE, width=2)
            self.ga.line("sadf", (ex, ey), (ex-0.05, ey+0.05), color=WHITE, width=2)

            from irlc.utils.graphics_util_pygame import Object
            if self.action is not None:
                for a in range(self.action):
                    self.truck = Object(file="truck.jpg", graphics=self.ga, image_width=0.25 * scale)
                    self.truck.move_center_to_xy(self.x + 1 + 0.2 + a * 0.2, 0.6 + dh)
                    self.truck.flip()
                    self.truck.blit(self.ga.surf)


        if self.w is not None:
            self.ga.text("asdf", (self.x + 1.5, dh+0.05), WHITE, contents=f"w_{self.k} = {self.w}", size=12, style="bold",
                         anchor="c")
            for w in range(self.w):
                self.customer = Object(file="customer.jpg", graphics=self.ga, image_width=0.25*scale)
                self.customer.move_center_to_xy(self.x + 1 + 0.2 + w * 0.2, 0.3 + dh)
                self.customer.blit(self.ga.surf)

        if self.reward is not None:
            self.ga.text("asdf", (self.x + 1.5, dh-0.02), WHITE, contents=f"reward = {self.reward}", size=12, style="bold",
                         anchor="c")
