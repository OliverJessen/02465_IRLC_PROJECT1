from __future__ import annotations

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from gymnasium import logger
from irlc.ex01.agent import Agent
import time
import sys
import gymnasium as gym
import os
import asyncio
import numpy as np

HUMAN_REQUEST_RESET = 'reset the environment'

try:
    # Using this backend apparently clash with scientific mode. Not sure why it was there in the first place so
    # disabling it for now.
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

try:
    import pygame
    from pygame.event import Event
except ImportError as e:
    logger.warn('failed to import pygame. Many of the interactive visualizations will not work: %s' % str(e))
    Event = None

class AgentWrapper(Agent):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def pi(self, state, k, info=None):
        return self.agent.pi(state, k, info)
        # return self.env.step(action)

    def train(self, *args, **kwargs):
        return self.agent.train(*args, **kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agent)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.agent.unwrapped

PAUSE_KEY = ord('p')
SPACEBAR = "_SPACE_BAR_PRESSED_"
HUMAN_DEMAND_RESET = "reset env."

async def _webassembly_interactive(env, agent, autoplay=False):
    from pygame import gfxdraw
    import types

    if not hasattr(agent, "reset"):
        from irlc.lectures.lec10.utils import agent_reset
        agent.reset = types.MethodType(agent_reset, agent)


    def aapolygon(surface, points, color):
        pygame.draw.polygon(surface, color, points)

    def filled_polygon(surface, points, color):
        pygame.draw.polygon(surface, color, points, width=0)

    def aacircle(surface, x, y, r, color):
        pygame.draw.circle(surface, color, (x, y), r, width=1)

    def filled_circle(surface, x, y, r, color):
        pygame.draw.circle(surface, color, (x, y), r, width=0)

    def hline(surface, x1, x2, y, color):
        pygame.draw.line(surface, color,  (x1, y), (x2, y) )

    gfxdraw.aapolygon = aapolygon
    gfxdraw.filled_polygon = filled_polygon
    gfxdraw.aacircle = aacircle
    gfxdraw.filled_circle = filled_circle
    gfxdraw.hline = hline


    # from irlc.utils.player_wrapper import AsyncPlayerWrapperPygame
    agent = AsyncPlayerWrapperPygame(agent, env, autoplay=autoplay)

    s, info = env.reset()
    k = 0
    await asyncio.sleep(0.2)  # set up graphics.
    running = True
    step = 0
    while running:
        pi_action = agent.pi(s, k, info)
        a = None
        truncated = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting the environment")
                    truncated = True

                a = agent._resolve_event_into_action(event=event, pi_action=pi_action, info=info)
                if a is not None:
                    break
            if agent.human_demand_autoplay:
                a = pi_action
                break
            if a is not None:
                break
            await asyncio.sleep(0.05)

        # if not isinstance(a, np.ndarray) and isinstance(a, str) and a == HUMAN_DEMAND_RESET:
        #     s, info = env.reset()
        #     continue
        if not truncated:
            sp, reward, done, truncated, info_sp = (await env.async_step(a)) if hasattr(env, 'async_step') else env.step(a)
            agent.train(s, a, reward, sp, done=done, info_s=info, info_sp=info_sp)

        step = step + 1
        k = k + 1
        if done or truncated:
            sp, info_sp = env.reset()
            k = 0

        s = sp
        info = info_sp
        await asyncio.sleep(0.01)  # release loop.


class PlayWrapperPygame(AgentWrapper):
    render_after_train = True
    ACTION_FORCE_RESET = "force reset of environment."

    def __init__(self, agent : Agent, env : gym.Env, keys_to_action=None, autoplay=False):
        super().__init__(agent, env)
        if keys_to_action is None:
            if hasattr(env, 'get_keys_to_action'):
                keys_to_action = env.get_keys_to_action()
            elif hasattr(env, "env") and hasattr(env.env, 'get_keys_to_action'):
                keys_to_action = env.env.get_keys_to_action()
            elif hasattr(env.unwrapped, 'get_keys_to_action'):
                keys_to_action = env.unwrapped.get_keys_to_action()
            else:
                keys_to_action = dict()
                # print(env.spec.id +" does not have explicit key to action mapping, please specify one manually")
                # assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                #               "please specify one manually"
                # keys_to_action = dict()
        self.keys_to_action = keys_to_action
        self.env = env
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.human_agent_action = -1
        self.human_demand_autoplay = autoplay

        # Now for the more fucky stuff. Collect internal statistics and such.
        self.env._interactive_data = dict(trajectories=[], in_terminal_state=False)
        # Now fix the train function
        train2 = agent.train
        reset2 = env.reset

        def reset_(**kwargs):
            from irlc.ex01.agent import Trajectory
            s, info = reset2(**kwargs)
            env._interactive_data['trajectories'].append(Trajectory(state=[s], time=[0], action=[], reward=[], env_info=[info]))
            env._interactive_data['in_terminal_state'] = False
            return s, info

        def train_(s, a, r, sp, done, info_s, info_sp):
            if not isinstance(a, str) or a != PlayWrapperPygame.ACTION_FORCE_RESET:
                train2(s, a, r, sp, done, info_s, info_sp)
                env._interactive_data['trajectories'][-1].state.append(sp)
                env._interactive_data['trajectories'][-1].reward.append(r)
                env._interactive_data['trajectories'][-1].action.append(a)
                env._interactive_data['trajectories'][-1].env_info.append(info_sp)

                if done:
                    env._interactive_data['in_terminal_state'] = True
                    env._interactive_data['avg_reward_per_episode'] = np.mean( [sum(t.reward) for t in env._interactive_data['trajectories']] )
                    env._interactive_data['completed_episodes'] = len(env._interactive_data['trajectories'] )

            if self.render_after_train: # always true.
                env.render()


        step2 = env.step
        def step_(a):
            if isinstance(a, str) and a == PlayWrapperPygame.ACTION_FORCE_RESET:
                print("Resetting interaction data.")
                self.env._interactive_data = dict(trajectories=[], in_terminal_state=False)
                return "done", 3, True, True, {}
            else:
                sp, r, terminated, truncated, info = step2(a)
            return sp, r, terminated, truncated, info


        agent.train = train_
        env.agent = agent
        env.unwrapped.agent = agent

        env.reset = reset_
        env.step = step_


    def _resolve_event_into_action(self, event : pygame.event.Event, pi_action, info : dict):
        # assert False # Unclear if used.
        if event.type == pygame.QUIT:
            if hasattr(self, 'env'):
                self.env.close()
            time.sleep(0.1)
            pygame.display.quit()
            time.sleep(0.1)
            pygame.quit()
            time.sleep(0.1)
            sys.exit()

        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            # if event.key == pygame.K_f:
            #     print("Pressing f")
            #     self.env.render()
            #     return None
            if event.key == pygame.K_SPACE:
                a = pi_action
                return a
            elif (event.key,) in self.keys_to_action:
                a = self.keys_to_action[(event.key,)]
                if info is not None and 'mask' in info:
                    from irlc.utils.common import DiscreteTextActionSpace
                    if isinstance(self.env.action_space, DiscreteTextActionSpace):
                        aint = self.env.action_space.actions.index(a)
                    else:
                        aint = a

                    if info['mask'][aint] == 0:
                        # The action was masked. This means that this action is unavailable, and we should select another.
                        # The default is to select one of the available actions from the mask.
                        from irlc.pacman.gamestate import GameState
                        from irlc.pacman.pacman_environment import PacmanEnvironment
                        if isinstance(self.env, PacmanEnvironment):
                            a = "Stop"
                        else:
                            a = info['mask'].argmax()
                            if isinstance(self.env.action_space, DiscreteTextActionSpace):
                                a = self.env.action_space.actions[a]
                    return a
                else:
                    return a
            elif event.key == pygame.K_r:
                print("Pressing r")
                if hasattr(self, 'reset'):
                    self.reset()
                return PlayWrapperPygame.ACTION_FORCE_RESET

            elif event.key == pygame.K_f:
                print("Pressing f")
                self.env.render()

            elif event.key == pygame.K_p:
                # unpause
                # print("Unpausing game")
                self.human_demand_autoplay = not self.human_demand_autoplay
                # print(f"Unpausing game {self.human_demand_autoplay=}")

                if self.human_demand_autoplay:
                    # print("Returning", pi_action)
                    a = pi_action
                    return a
            else:
                # try to pass event on to the game.
                if hasattr(self.env, 'keypress'):
                    self.env.keypress(event)
        return None

    def pi(self, state, k, info=None):
        # print("Entering pi")
        pi_action = super().pi(state, k, info) # make sure super class pi method is called in case it has side effects.
        a = None
        while True:
            # print(f"while loop {self.human_demand_autoplay=}")
            for event in pygame.event.get():
                # print("resolving", event)
                a = self._resolve_event_into_action(event, pi_action, info)
                # print("Resolved", a)
                if a is not None:
                    # print("Breaking", a)
                    break
                # if False:
                #     if event.type == pygame.QUIT:
                #         if hasattr(self, 'env'):
                #             self.env.close()
                #         time.sleep(0.1)
                #         pygame.display.quit()
                #         time.sleep(0.1)
                #         pygame.quit()
                #         time.sleep(0.1)
                #         sys.exit()
                #
                #     # checking if keydown event happened or not
                #     if event.type == pygame.KEYDOWN:
                #         if event.key == pygame.K_SPACE:
                #             a = pi_action
                #             break
                #         elif (event.key,) in self.keys_to_action:
                #             a = self.keys_to_action[(event.key,)]
                #             if info is not None and 'mask' in info:
                #                 from irlc.utils.common import DiscreteTextActionSpace
                #                 if isinstance(self.env.action_space, DiscreteTextActionSpace):
                #                     aint = self.env.action_space.actions.index(a)
                #                 else:
                #                     aint = a
                #
                #                 if info['mask'][aint] == 0:
                #                     # The action was masked. This means that this action is unavailable, and we should select another.
                #                     # The default is to select one of the available actions from the mask.
                #                     a = info['mask'].argmax()
                #                     if isinstance(self.env.action_space, DiscreteTextActionSpace):
                #                         a = self.env.action_space.actions[a]
                #                 break
                #             else:
                #                 break
                #         elif event.key == pygame.K_r:
                #             print("Pressing r")
                #             if hasattr(self, 'reset'):
                #                 return PlayWrapperPygame.ACTION_FORCE_RESET
                #             #
                #             #     self.reset()
                #             #     self.env.reset()
                #             #
                #             # self.env.render()
                #         elif event.key == pygame.K_f:
                #             print("Pressing f")
                #             self.env.render()
                #
                #         elif event.unicode == 'p':
                #             # unpause
                #             self.human_demand_autoplay = not self.human_demand_autoplay
                #             break
                #         else:
                #             # try to pass event on to the game.
                #             if hasattr(self.env, 'keypress'):
                #                 self.env.keypress(event)

            if self.human_demand_autoplay:
                a = pi_action

            if a is not None:
                try:
                    from irlc.pacman.gamestate import GameState
                    if isinstance(state, GameState):
                        if a not in state.A():
                            a = "Stop"
                except Exception as e:
                    pass
                # print(f"{a=}")
                return a
            time.sleep(0.1)




class AsyncPlayerWrapperPygame(PlayWrapperPygame):
    # render_after_train = False

    def pi(self, state, k, info=None):
        pi_action = self.agent.pi(state, k, info)  # make sure super class pi method is called in case it has side effects.
        return pi_action

    def _post_process_action_for_visualization(self):
        pass


def interactive(env : gym.Env, agent: Agent, autoplay=False) -> (gym.Env, Agent):
    """
    This function is used for visualizations. It can

    - Allow you to input keyboard commands to an environment
    - Allow you to save results
    - Visualize reinforcement-learning agents in the gridworld environment.

    by adding a single extra line ``env, agent = interactive(env,agent)``.
    The following shows an example:

        >>> from irlc.gridworld.gridworld_environments import BookGridEnvironment
        >>> from irlc import train, Agent, interactive
        >>> env = BookGridEnvironment(render_mode="human", zoom=0.8) # Pass render_mode='human' for visualization.
        >>> env, agent = interactive(env, Agent(env))               # Make the environment interactive. Note that it needs an agent.
        >>> train(env, agent, num_episodes=2)                     # You can train and use the agent and environment as usual.
        >>> env.close()

    It also enables you to visualize the environment at a matplotlib figure or save it as a pdf file using ``env.plot()`` and ``env.savepdf('my_file.pdf)``.

    All demos and figures in the notes are made using this function.

    :param env: A gym environment (an instance of the ``Env`` class)
    :param agent: An agent (an instance of the ``Agent`` class)
    :param autoplay: Whether the simulation should be unpaused automatically
    :return: An environment and agent which have been slightly updated to make them interact with each other. You can use them as usual with the ``train``-function.
    """
    from PIL import Image # Let's put this one here in case we run the code in headless mode.

    agent = PlayWrapperPygame(agent, env, autoplay=autoplay)

    pass

    def plot():
        env.render_mode, rmt = 'rgb_array', env.render_mode
        frame = env.render()
        env.render_mode = rmt
        im = Image.fromarray(frame)
        plt.imshow(im)
        plt.axis('off')
        plt.axis('off')
        plt.tight_layout()

    def savepdf(file):
        env.render_mode, rmt = 'rgb_array', env.render_mode
        frame = env.render()
        env.render_mode = rmt

        im = Image.fromarray(frame)
        snapshot_base = file
        if snapshot_base.endswith(".png"):
            sf = snapshot_base[:-4]
            fext = 'png'
        else:
            fext = 'pdf'
            if snapshot_base.endswith(".pdf"):
                sf = snapshot_base[:-4]
            else:
                sf = snapshot_base

        sf = f"{sf}.{fext}"
        dn = os.path.dirname(sf)
        if len(dn) > 0 and not os.path.isdir(dn):
            os.makedirs(dn)
        print("Saving snapshot of environment to", os.path.abspath(sf))
        if fext == 'png':
            im.save(sf)
            from irlc import _move_to_output_directory
            _move_to_output_directory(sf)
        else:
            plt.figure(figsize=(16, 16))
            plt.imshow(im)
            plt.axis('off')
            plt.tight_layout()
            from irlc import savepdf
            savepdf(sf, verbose=True)
            plt.show()
    env.plot = plot
    env.savepdf = savepdf
    return env, agent


def main():
    from irlc.gridworld.gridworld_environments import BookGridEnvironment  
    from irlc import Agent
    env = BookGridEnvironment(render_mode="human", zoom=0.8)  # Pass render_mode='human' for visualization.
    env, agent = interactive(env, Agent(env))  # Make th
    env.reset()     # We always need to call reset
    env.plot()      # Plot the environment.
    env.close()  

    # Interaction with a random agent.
    from irlc.gridworld.gridworld_environments import BookGridEnvironment 
    from irlc import train, Agent
    env = BookGridEnvironment(render_mode="human", zoom=0.8) # Pass render_mode='human' for visualization.
    env, agent = interactive(env, Agent(env))               # Make the environment interactive. Note that it needs an agent.
    train(env, agent, num_episodes=100)                      # You can train and use the agent and environment as usual. 
    env.close()


if __name__ == "__main__":
    main()
