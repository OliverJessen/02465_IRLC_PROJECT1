# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import pygame.draw
import pygame
from irlc.ex07.bandits import StationaryBandit
import time
import numpy as np
from irlc.pacman.pacman_resources import WHITE, BLACK, Ghost


class BinaryBandit(StationaryBandit):
    def reset(self):
        # self.q_star = np.random.rand(self.k) + self.q_star_mean
        self.q_star = np.ones(self.k)/3
        self.q_star[np.random.randint(self.k)] *= 2
        self.optimal_action = np.argmax(self.q_star)
        self.previous_action = None
        self.reward = None
        return None, {}

    def bandit_step(self, a):
        """ Return the reward/regret for action a for the simple bandit. Use self.q_star (see reset) """
        reward = np.random.rand() < self.q_star[a]
        regret = self.q_star[self.optimal_action] - self.q_star[a]
        self.previous_action = a
        self.reward = reward
        return reward, regret


class GraphicalBandit(BinaryBandit):
    viewer = None
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': 20}

    def get_keys_to_action(self):
        return {(pygame.K_0,):0, (pygame.K_1,): 1, (pygame.K_2,): 2, (pygame.K_3,): 3, (pygame.K_4,): 4,
                (pygame.K_5,): 5, (pygame.K_6,): 6, (pygame.K_7,): 7, (pygame.K_8,): 8, (pygame.K_9,): 9,
                # (pygame.K_0,):0
                }

    def __init__(self, *args, render_mode='human', frames_per_second=None, **kwargs):
        self.previous_reward, self.previous_action, self.agent = None, None, None
        super().__init__( *args, **kwargs)
        self.render_mode = render_mode
        self.viewer = None
        self.show_q_star = False
        self.show_q_ucb = False
        self.frames_per_second = frames_per_second

        print("press q to show true q values and u to show UCB upper bounds.")

    def reset(self):
        s, info = super().reset()
        if hasattr(self, 'agent'):
            if hasattr(self.agent, 'Q'): del self.agent.Q
            if hasattr(self.agent, 'N'): del self.agent.N

        self.render()

        return s, info

    def step(self, action):
        o = super().step(action)
        self.previous_action = action
        self.previous_reward = o[1]
        self.render()
        return o

    def keypress(self, key):
        # print(key)
        if key.unicode == 'q':
            self.viewer.show_q_star = not self.viewer.show_q_star

        if key.unicode == 'u':
            self.viewer.show_q_ucb = not self.viewer.show_q_ucb
        self.render()

    def render(self, mode='human', agent=None, prev_action=None, reward=None):
        if self.viewer is None:
            self.viewer = BanditViewer(self, frames_per_second=self.frames_per_second)
        self.viewer.update(self.agent, self.previous_action, self.previous_reward)
        return self.viewer.blit(render_mode=self.render_mode) #(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.viewer.close()

class BanditViewer:
    scale = 400  # Scale of a single bar.
    width = 0.4 * scale  # with of a bar.
    bar_height = scale

    def __init__(self, bandit, frames_per_second=None):
        bin_bandit = isinstance(bandit, BinaryBandit)
        if bin_bandit:
            ymin = 0 - 0.6 * self.scale
            ymax = (1 + 0.4)* self.scale
        else:
            ymin = (min(bandit.q_star) - 0.6)*self.scale
            ymax = (max(bandit.q_star) + 0.4)*self.scale

        xmin = -self.width
        xmax = (bandit.k * self.width  + self.width)
        # super().__init__(screen_width=1300, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        from irlc.utils.graphics_util_pygame import GraphicsUtilGym
        dx = xmax-xmin
        dy = ymax-ymin

        # screen_width = 1300, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax
        self.ga = GraphicsUtilGym()
        screen_width = 1300
        self.ga.begin_graphics(screen_width, dy * (screen_width / dx), local_xmin_xmax_ymin_ymax=(xmin, xmax, ymax, ymin), frames_per_second=frames_per_second)
        self.bandit = bandit
        self.ghost = Ghost(self.ga, scale=100)

        # self.ghost.surf = pygame.transform.scale(self.ghost.surf, (self.ghost.surf.get_width()*0.4, self.ghost.surf.get_height()*0.4) )
        # self.ghost.rect = self.ghost.surf.get_rect()
        # self.ghost.rect.x = self.width / 2
        # self.ghost.rect.y = -self.scale

        self.last_action = None
        self.agent = None
        self.last_reward = None

        self.show_q_star = False
        self.show_q_ucb = False
        # self.ghost.group.scale(self.width*0.4)
        # self.ghost.group.translate(self.width / 2, -self.scale)

    def close(self):
        self.ga.close()

    def blit(self, render_mode='human'):
        return self.ga.blit(render_mode=render_mode)

    def master_render(self):
        self.ga.draw_background()
        # batch = pyglet.graphics.Batch()
        # self.ghosts = pyglet.graphics.Batch()
        # return
        self.q_star = []
        self.q_ucb_upper = []
        self.Qs = []
        # bgg = OrderedGroup(-1)
        # self.bg = [shapes.Rectangle(xmin, ymin, xmax-xmin, ymax-ymin, color=(0,0,0), batch=batch, group=bgg)]

        dd = self.width / 30
        self.text_n = []
        fz = int(self.width / 6)

        dw = self.width / 4

        # group = OrderedGroup(1)
        for i in range(self.bandit.k):
            x = i * self.width
            # print(x)
            # from pyglet import shapes
            # from pyglet.shapes import Rectangle
            # self.bg.append(shapes.Rectangle(20, 30, 100,100, color=(100,10,10), batch=batch,group=group))

            # pygame.draw.rect()
            # self.ga.polygon()
            # self.ga.rectangle(self.ga.surf, WHITE, pygame.Rect)
            self.ga.rectangle(WHITE, x-dd, -dd, self.width/2+dd*2, self.bar_height+dd*2, border=0)
            self.ga.rectangle(BLACK, x, 0, self.width/2, self.bar_height, border=0)

            if self.agent is not None and hasattr(self.agent, 'Q'):
                # height =
                # print(self.agent.Q[i] * self.bar_height)
                self.ga.rectangle((150, 200, 150), x, 0, self.width / 2, self.agent.Q[i] * self.bar_height)  # q-values.

            # self.ga.rectangle((150,200,150), x, 0, self.width / 2, 0, ) # q-values.

            self.ga.text("sadf", (x + self.width / 4, self.bar_height + dw * 2), WHITE, contents=f"Arm  {i}",
                         size=fz,
                         style='bold')
            if self.agent is not None:
                self.ga.text("sadf", (x + self.width / 4, self.bar_height + dw), WHITE, contents= f"N = {int(self.agent.N[i] if hasattr(self.agent, 'N') else 0)}", size=fz,
                             style='bold')



            # return
            # continue
            # return
            # self.bg.append(shapes.Rectangle(x-dd, -dd, self.width/2+dd*2, self.bar_height+dd*2, color=WHITE,batch=batch,group=group))
            # self.bg.append(shapes.Rectangle(x, 0, self.width/2, self.bar_height, color=BLACK, batch=batch,group=group))
            # return
            # self.Qs.append(shapes.Rectangle(x, 0, self.width/2, .0, color=(150,200,150), batch=batch,group=group))

            # q_star = shapes.Rectangle(x, 0, self.width / 2, dd, color=WHITE, batch=batch,group=group)
            # self.q_star_visible = False
            if self.show_q_star:

                y =  self.bandit.q_star[i] * self.bar_height
                self.ga.rectangle(WHITE, x, y, self.width / 2, dd)  # q-star
                # print(x, y)

            # q_star.visible = False
            # self.q_star.append(q_star)
            # self.q_ucb_visible = False
            if self.show_q_ucb:
                from irlc.ex07.ucb_agent import UCBAgent
                # if :
                if (hasattr(self.agent, 'c') or isinstance(self.agent, UCBAgent)) and hasattr(self.agent, 'N'):  # Required if reset has not been called.
                    t = (sum(self.agent.N) + 1e-8)
                    ub = self.agent.Q + self.agent.c * np.sqrt(np.log(t + 1) / (self.agent.N + 1e-8))
                    self.ga.rectangle((200, 0, 0), x, ub[i] * self.bar_height, self.width / 2, dd)
                # q_ucb = shapes.Rectangle(x, -1000, self.width / 2, dd, color=(200, 0, 0), batch=batch,group=group)
            # q_ucb.visible = False
            # self.q_ucb_upper.append(q_ucb)

            # self.q_star = []
            # for i, b in enumerate(self.q_star):
            #     continue
            #     b.y = self.bandit.q_star[i] * self.bar_height
            # print(ub)
            # self.bg.append(Label(f"Arm  {i}", font_name='Arial', x=x+self.width/4, y=self.bar_height + dw*2, anchor_x='center', bold=True, color=(255, 255, 255, 255), anchor_y='center', font_size=fz, batch=batch, group=group))
            # self.text_n.append(Label(f"N = {0}", font_name='Arial', x=x+self.width/4, y=self.bar_height + dw, anchor_x='center', bold=True, color=(255, 255, 255, 255), anchor_y='center', font_size=fz, batch=batch, group=group))
        if self.agent is not None:
            self.ga.text("sadf", (self.width / 2, self.bar_height + dw * 3), WHITE,
                         contents=f"{self.agent.method if hasattr(self.agent, 'method') else ''}", size=fz,
                         style='bold', anchor='c')
        reward = self.last_reward
        action = self.last_action
        self.ghost.set_direction(self.ghost.rand_eyes()) # Random eyes.
        if reward is not None and action is not None:
            if reward <= 0:
                self.ghost.kill()
            else:
                self.ghost.resurrect()
            last_outcome_x =  (action+0.25)*self.width
            last_outcome_y = reward
            self.ga.circle("cc", (last_outcome_x, last_outcome_y), self.width / 10, fillColor=WHITE, outlineColor=None)

            # self.last_outcome.x = (action+0.25)*self.width
            # self.last_outcome.y = reward
        else:
            pass
            # self.last_outcome.x = -self.scale

        if action is None:
            action = -0.5

        y2 = -0.45*self.width - 0.25 * self.width
        x2 = (action+.25) * self.width
        self.ghost.set_position(x2, y2)
        self.ghost.render()
        return
        # for i, b in enumerate(self.q_star):
        #     b.y = self.bandit.q_star[i]* self.bar_height
        #
        #     from irlc.ex08.ucb_agent import UCBAgent
        #     if isinstance(agent, UCBAgent) and hasattr(agent, 'N'): # Required if reset has not been called.
        #         t = sum(agent.N)
        #         ub = agent.Q + agent.c * np.sqrt(np.log(t + 1) / (agent.N + 1e-8))
        #
        #         for i, b in enumerate(self.q_ucb_upper):
        #             b.y = ub[i]* self.bar_height
        #
        #         # print(ub)
        #     s = 234

        # self.last_outcome = shapes.Circle(-self.scale, 0, self.width/10, color=WHITE, batch=batch,group=group)
        # self.batch = batch

    def update(self, agent, action, reward):
        self.agent = agent
        self.last_action = action
        self.last_reward = reward
        self.master_render()

        # return
        # # return
        #
        # # if action is not None:
        # #     y2 = -0.45*self.width
        # #     x2 = (action+.25) * self.width
        # #     self.ghost.set_position(x2, y2)
        #
        #
        # # dd = self.width / 30
        # self.text_n = []
        # # group = OrderedGroup(1)
        # for i in range(self.bandit.k):
        #     x = i * self.width
        #     # print(x)
        #     # from pyglet import shapes
        #     # from pyglet.shapes import Rectangle
        #     # self.bg.append(shapes.Rectangle(20, 30, 100,100, color=(100,10,10), batch=batch,group=group))
        #
        #     # pygame.draw.rect()
        #     # self.ga.polygon()
        #     # self.ga.rectangle(self.ga.surf, WHITE, pygame.Rect)
        #     self.ga.rectangle(WHITE, x - dd, -dd, self.width / 2 + dd * 2, self.bar_height + dd * 2, border=0)
        #     self.ga.rectangle(BLACK, x, 0, self.width / 2, self.bar_height, border=0)
        #
        #
        #     dw = self.width / 4
        #     fz = int(self.width / 6)
        #
        #     # for i, b in enumerate(self.Qs):
        #     if agent is not None:
        #         if hasattr(agent, 'Q'):
        #             height = agent.Q[i] * self.bar_height
        #             self.ga.rectangle((150, 200, 150), x, 0, self.width / 2, height)  # q-values.
        #
        #         if hasattr(agent, 'N'):
        #             nlabel = f"N = {int(agent.N[i])}"
        #             self.ga.text("sadf", (x + self.width / 4, self.bar_height + dw), WHITE, contents=nlabel,
        #                          size=fz,
        #                          style='bold', anchor='c')
        #
        #             # self.text_n[i].text =
        #
        #     self.ga.text("sadf", (x + self.width / 4, self.bar_height + dw * 2), WHITE, contents=f"Arm  {i}",
        #                  size=fz,
        #                  style='bold', anchor='c')
        #
        # return


    # def draw(self):
    #     self.batch.draw()
    #     self.ghosts.draw()


if __name__ == "__main__":
    env = GraphicalBandit(10, render_mode='human')
    from irlc import train
    from irlc.ex07.ucb_agent import UCBAgent
    # from irlc.utils.player_wrapper import PlayWrapper
    from irlc import interactive
    # agent = BasicAgent(env, epsilon=0.1)
    agent = UCBAgent(env)

    # env = VideoMonitor(env, agent=agent)
    env, agent = interactive(env, agent)

    # agent = PlayWrapper(agent, env)

    t0 = time.time()
    n = 500
    stats, _ = train(env, agent, max_steps=n, num_episodes=10, return_trajectory=False, verbose=False)
    tpf = (time.time()-t0)/ n
    print("tpf", tpf, 'fps', 1/tpf)
    env.close()
