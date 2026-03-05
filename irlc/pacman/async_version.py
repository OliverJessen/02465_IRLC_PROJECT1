# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.pacman.pacman_graphics_display import PacmanGraphics, FirstPersonPacmanGraphics
import asyncio

class AsyncPacmanGraphics(PacmanGraphics):
    async def update(self, newState, animate=False, ghostbeliefs=None, path=None, visitedlist=None):
        agentIndex = newState.data._agentMoved
        agentState = newState.data.agentStates[agentIndex]
        if self.agentImages[agentIndex][0].isPacman != agentState.isPacman: self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        if animate:
            if agentState.isPacman:
                await self.animatePacman(agentState, prevState, prevImage, state=newState, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
            else:
                self.moveGhost(agentState, agentIndex, prevState, prevImage)

        self.agentImages[agentIndex] = (agentState, prevImage)

        if newState.data._foodEaten != None:
            self.removeFood(newState.data._foodEaten, self.food)
        if newState.data._capsuleEaten != None:
            self.removeCapsule(newState.data._capsuleEaten, self.capsules)

        if 'ghostDistances' in dir(newState):
            self.infoPane.updateGhostDistances(newState.data.ghostDistances)
        self.master_render(newState, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)

    async def animatePacman(self, pacman, prevPacman, image, nframe=1, frames=4, state=None, ghostbeliefs=None, path=None, visitedlist=None):
        if self.frameTime < 0:
            print('Press any key to step forward, "q" to play')
        if self.frameTime > 0.01 or self.frameTime < 0:
            fx, fy = self.getPosition(prevPacman)
            px, py = self.getPosition(pacman)
            for nframe in range(1,int(frames) + 1):
                pos = px*nframe/frames + fx*(frames-nframe)/frames, py*nframe/frames + fy*(frames-nframe)/frames
                self.movePacman(pos, self.getDirection(pacman), image, pacman=pacman)
                pacman.draw_extra['delta_xy'] = (pos[0]-px, pos[1]-py)
                await asyncio.sleep(self.frameTime/frames)
                self.master_render(state, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
                self.blit(render_mode='human')
        else:
            self.movePacman(self.getPosition(pacman), self.getDirection(pacman), image, pacman=pacman)
    pass

class AsyncPacmanEnvironment(PacmanEnvironment):
    def _private_make_graphics(self):
        if self.first_person_graphics:
            self.graphics_display = FirstPersonPacmanGraphics(self.game.state, self.options_zoom, showGhosts=True,
                                                              frameTime=self.options_frametime,
                                                              ghostbeliefs=self.ghostbeliefs)
        else:
            self.graphics_display = AsyncPacmanGraphics(self.game.state, self.options_zoom, frameTime=self.options_frametime,
                                                   method=self.method)

    async def async_step(self, action):
        r_ = self.game.state._unsafe_getScore()
        done = False
        if action not in self.state.A():
            raise Exception(f"Agent tried {action=} available actions {self.state.A()}")

        # Let player play `action`, then let the ghosts play their moves in sequence.
        for agent_index in range(len(self.game.agents)):
            a = self.game.agents[agent_index].getAction(self.game.state) if agent_index > 0 else action
            self.game.state = self.game.state.f(a)
            self.game.rules.process(self.game.state, self.game)

            if self.graphics_display is not None and self.animate_movement and agent_index == 0:
                await self.graphics_display.update(self.game.state, animate=self.animate_movement, ghostbeliefs=self.ghostbeliefs, path=self.path, visitedlist=self.visitedlist)
            done = self.game.gameOver or self.game.state.is_won() or self.game.state.is_lost()
            if done:
                break
        reward = self.game.state._unsafe_getScore() - r_
        return self.state, reward, done, False, {'mask': self.action_space._make_mask(self.state.A())}
