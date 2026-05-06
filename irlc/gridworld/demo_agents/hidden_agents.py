# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
import numpy as np
from irlc import TabularAgent # , PlayWrapper, VideoMonitor, train
from irlc.ex08.mdp_warmup import value_function2q_function


class ValueIterationAgent2(TabularAgent):
    def __init__(self, env, gamma=.99, epsilon=0, theta=1e-5, only_current_state=False):
        self.v = defaultdict(lambda: 0)
        self.steps = 0
        self.mdp = env.mdp
        self.only_current_state = only_current_state
        super().__init__(env, gamma, epsilon=epsilon)

    def pi(self, s, k, info=None): 
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")
        return self.random_pi(s) if np.random.rand() < self.epsilon else a

    @property
    def label(self):
        label = f"Value iteration after {self.steps} steps"
        return label

    def v2Q(self, s): # used for rendering right now
        return value_function2q_function(self.mdp, s, self.gamma, self.v)

    def train(self, s, a, r, sp, done=False, info_sp=None):
        delta = 0
        v2 = {}
        for s in self.env.P.keys():
            v, v2[s] = self.v[s], max(value_function2q_function(self.mdp, s, self.gamma, self.v).values()) if len(self.mdp.A(s)) > 0 else 0
            delta = max(delta, np.abs(v - self.v[s]))

        self.v = v2

        for s in self.mdp.nonterminal_states:
            for a in self.mdp.A(s):
                self.Q[s,a] = self.v2Q(s)[a]

        self.delta = delta
        self.steps += 1

    def __str__(self):
        return f"VIAgent_{self.gamma}"


class PolicyEvaluationAgent2(TabularAgent):
    def __init__(self, env, mdp=None, gamma=0.99, steps_between_policy_improvement=10, only_update_current=False):
        if mdp is None:
            mdp = env.mdp
        self.mdp = mdp
        self.v = defaultdict(lambda: 0)
        self.imp_steps = 0
        self.steps_between_policy_improvement = steps_between_policy_improvement
        self.steps = 0
        self.policy = {}
        self.only_update_current = only_update_current
        for s in mdp.nonterminal_states:
            self.policy[s] = {}
            for a in mdp.A(s):
                self.policy[s][a] = 1/len(mdp.A(s))
        super().__init__(env, gamma)

    def reset(self):
        self.v = defaultdict(lambda: 0)

    def pi(self, s,k, info=None):  
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")
        return np.random.choice(a, p=pa)

    def v2Q(self, s):  # used for rendering right now
        return value_function2q_function(self.mdp, s, self.gamma, self.v)

    @property
    def label(self):
        if self.steps_between_policy_improvement is None:
            label = f"Policy evaluation after {self.steps} steps"
        else:
            dd = self.steps % self.steps_between_policy_improvement == 0
            # print(dd)
            label = f"PI after {self.steps} steps/{self.imp_steps-dd} policy improvements"
        return label

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        if not self.only_update_current:
            v2 = {}
            for s in self.mdp.nonterminal_states:
                q = value_function2q_function(self.mdp, s, self.gamma, self.v)
                if len(q) == 0:
                    v2[s] = 0
                else:
                    v2[s] = sum( [qv * self.policy[s][a] for a, qv in q.items()] )


            for s in self.mdp.nonterminal_states:
                for a,q in self.v2Q(s).items():
                    self.Q[s,a] = q

            for k, v in v2.items():
                self.v[k] = v2[k]

        else:
            # Only update Q-value in current state:
            Q_ = 0
            # print(a)

            for (sp, r), p in self.mdp.Psr(s, a).items():
                Q_ += p*(r + (0 if self.mdp.is_terminal(sp) else sum([self.Q[sp, ap]*pa for ap, pa in self.policy[sp].items()]) ))

                # Q_ += p * (r + (0 if self.mdp.is_terminal(sp) else sum(
                #     [self.Q[sp, ap] * pa for ap, pa in self.policy[sp].items()])))


            self.Q[s, a] = Q_

            v_ = 0
            for a in self.mdp.A(s):
                for (sp, r), p in self.mdp.Psr(s, a).items():
                    v_ += self.policy[s][a] * (self.v[sp] * self.gamma + r)*p
            self.v[s] = v_


        if self.steps_between_policy_improvement is not None and (self.steps+1) % self.steps_between_policy_improvement == 0:
            self.policy = {}
            for s in self.mdp.nonterminal_states:
                q = value_function2q_function(self.mdp, s, self.gamma, self.v)
                if len(q) == 0:
                    continue
                a_ = max(q, key=q.get)  # optimal action
                self.policy[s] = {}
                for a in self.mdp.A(s):
                    self.policy[s][a] = 1 if q[a] == max(q.values()) else 0 #if a == a_ else 0

                n = sum(self.policy[s].values())
                for a in self.policy[s]:
                    self.policy[s][a] *= 1/n

            self.imp_steps += 1
        self.steps += 1

    def __str__(self):
        return f"PIAgent_{self.gamma}"



class ValueIterationAgent3(TabularAgent):
    def __init__(self, env, mdp=None, epsilon=0, gamma=0.99, steps_between_policy_improvement=10, only_update_current=False):
        if mdp is None:
            mdp = env.mdp
        self.mdp = mdp
        self.v = defaultdict(lambda: 0)
        self.imp_steps = 0
        self.steps_between_policy_improvement = steps_between_policy_improvement
        self.steps = 0
        self.policy = {}
        self.only_update_current = only_update_current
        self.v = defaultdict(float)
        for s in mdp.nonterminal_states:
            self.policy[s] = {}
            for a in mdp.A(s):
                self.policy[s][a] = 1/len(mdp.A(s))
        super().__init__(env, gamma, epsilon=epsilon)

    def reset(self):
        self.v = defaultdict(lambda: 0)
        self.Q.q_.clear()



    def pi(self, s,k, info=None):
        from irlc import Agent
        if np.random.rand() <self.epsilon:
            return Agent.pi(self, s, k=k, info=info)

        a, pa = zip(*self.policy[s].items())
        return np.random.choice(a, p=pa)


    def v2Q(self, s):  # used for rendering right now
        if not self.only_update_current:
            a,q =  self.Q.get_Qs(s)
            return {a_: q_ for a_, q_ in zip(a,q)}
        else:
            return value_function2q_function(self.mdp, s, self.gamma, self.v)


    def vi_q(self, s, a):
        Q_ = 0
        for (sp, r), p in self.mdp.Psr(s, a).items():
            if self.mdp.is_terminal(sp):
                QT = 0
            else:
                qvals = [self.Q[sp, a_] for a_ in self.mdp.A(sp)]
                QT = max(qvals) * (1-self.epsilon) + self.epsilon*np.mean(qvals)
            Q_ += p * (r + self.gamma * QT)
        return Q_

    @property
    def label(self):
        label = f"Value Iteration after {self.steps} steps"
        return label

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        s_ = s
        if not self.only_update_current:
            q_ = dict()
            for s in self.mdp.nonterminal_states:
                for a in self.mdp.A(s):
                    q_[s,a] = self.vi_q(s, a)
            for (s,a), q in q_.items():
                self.Q[s,a] = q
        else:
            # Only update Q-value in current state:
            # s = s_
            qq = value_function2q_function(self.mdp, s, self.gamma, self.v)
            self.v[s] = max(qq.values())
            self.Q[s, a] = self.vi_q(s,a)


        for s in self.mdp.nonterminal_states:
            # q = qs_(self.mdp, s, self.gamma, self.v)
            # if len(q) == 0:
            #     continue
            # a_ = max(q, key=q.get)  # optimal action
            self.policy[s] = {}
            qs = [self.Q[s,a] for a in self.mdp.A(s)]

            for a in self.mdp.A(s):
                self.policy[s][a] = 1 if self.Q[s,a] >= max(qs)-1e-6 else 0 #if a == a_ else 0
            S = sum(self.policy[s].values())
            for a in self.mdp.A(s):
                self.policy[s][a] = self.policy[s][a] / S
            if not self.only_update_current:
                self.v[s] = max([self.Q[s, a_] for a_ in self.mdp.A(s)])

        self.steps += 1

    def __str__(self):
        return f"PIAgent_{self.gamma}"
