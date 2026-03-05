# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex09.mc_agent import MCAgent
from irlc.ex08.rl_agent import TabularQ
from collections import defaultdict

class MCAgentResettable(MCAgent):
    def reset(self):
        return agent_reset(self)

def agent_reset(self):
    # General reset option. Wroks on many agents.
    attrs = ['returns_sum_S', 'returns_count_N', 'Q', 'v', 'Model', 'e']

    for attr in attrs:
        if hasattr(self, attr):
            at = getattr(self, attr)
            if isinstance(at, dict) or isinstance(at, defaultdict):
                at.clear()
            elif isinstance(at, list):
                at.clear()

    if hasattr(self, 'Q') and isinstance(self.Q, TabularQ):
        self.Q.q_.clear()
