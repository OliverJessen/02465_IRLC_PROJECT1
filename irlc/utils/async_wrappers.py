# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from gymnasium.wrappers import TimeLimit

from irlc.pacman.pacman_environment import PacmanWinWrapper




class AsyncPacmanWinWrapper(PacmanWinWrapper):
    async def async_step(self, action):
        observation, reward, done, truncated, info = await self.env.async_step(action)
        if self.env.unwrapped.game.state.is_won():
            reward = 1
        else:
            reward = 0
        return observation, reward, done, truncated, info


class AsyncTimeLimit(TimeLimit):
    # def __init__(s

    async def async_step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = await self.env.async_step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info




def _fix_webassembly_packages(yes_really_do_it=False):
    import importlib
    import os
    assert yes_really_do_it, "This function is for internal use for deploying webassembly projects. Don't use it in your base dir."

    spec = importlib.util.find_spec("sympy", None)
    base = os.path.dirname(spec.origin)
    testf = f"{base}/testing/__init__.py"
    if base.startswith("/data/data/"):
        # with open(testf, 'w') as f:
        #     f.write("# Nothingatall")
        # with open(f"{base}/testing/runtests.py", 'w') as f:
        #     f.write("# Nothingatall")

        fname = f"{base}/utilities/decorator.py"
        assert os.path.isfile(fname)
        code = open(fname, 'r').read()
        with open(fname, 'w') as f:
            # print(f"{fname=}")
            f.write(ncode := "\n".join([l for l in code.splitlines() if not l.startswith("from sympy.testing")]))

        code = open(fname := f"{base}/utilities/__init__.py", 'r').read()
        code = code.replace("from .timeutils import timed", "timed = lambda x: 3")
        with open(fname, 'w') as f:
            f.write(code)

        for fname in [f"{base}/core/parameters.py", f"{base}/matrices/utilities.py"]:
            code = open(fname, 'r').read()
            code = code.replace("from threading import local", "local = object")
            with open(fname, 'w') as f:
                f.write(code)

        # Fix timeit.
        code = open(fname := f"{base}/utilities/timeutils.py", 'r').read()
        code = code.replace("import timeit", "# REMOVED")
        with open(fname, 'w') as f:
            f.write(code)

        code = open(fname := f"{base}/testing/runtests.py", 'r').read()
        code = code.replace("from timeit import default_timer as clock", "# REMOVED")
        # DocTestFinder, DocTestRunner
        #
        # code = code.replace("import doctest as pdoctest", "# REMOVED")

        # code = code.replace("from doctest import DocTestFinder, DocTestRunner", "DocTestFinder, DocTestRunner = object, object")
        # code = code.replace("pdoctest._indent", "#REMOVED")
        # code = code.replace("import doctest", "# REMOVED")

        with open(fname, 'w') as f:
            f.write(code)
        print("Patched ok.")
        """NB. Remember to also patch Decimal by adding extra stuff like exceptions to the decimal-module which is masked by webassembly."""

    pass
