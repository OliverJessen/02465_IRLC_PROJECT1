# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

# Use GPU; If the drivers give you grief you can turn GPU off without a too big hit on performance in the cartpole task
USE_CUDA = torch.cuda.is_available()

USE_CUDA = False # No, we use CPU.
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
from irlc.ex13.dqn_network import DQNNetwork

class TorchNetwork(nn.Module,DQNNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001, hidden=30):
        nn.Module.__init__(self)
        DQNNetwork.__init__(self)
        self.env = env
        self.hidden = hidden
        self.actions = env.action_space.n
        self.build_model_()
        if trainable:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if USE_CUDA:
            self.cuda()

    def build_feature_network(self):
        num_observations = np.prod(self.env.observation_space.shape)
        return (nn.Linear(num_observations, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU())

    def build_model_(self):
        num_actions = self.env.action_space.n
        self.model = nn.Sequential(*self.build_feature_network(), nn.Linear(self.hidden,num_actions))

    def forward(self, s):
        s = Variable(torch.FloatTensor(s))
        s = self.model(s)
        return s

    def __call__(self, s):
        return self.forward(s).detach().numpy()

    def fit(self, s, target):
        q_value = self.forward(s)
        loss = (q_value - torch.FloatTensor(target).detach()).pow(2).sum(axis=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_Phi(self, source, tau=1):
        """
        Polyak adapt weights of this class given source:
        I.e. tau=1 means adopt weights in one step,
        tau = 0.001 means adopt very slowly, tau=1 means instant overwriting
        """
        state = self.state_dict()
        for k, wa in state.items():
            wb = source.state_dict()[k]
            state[k] = wa*(1 - tau) + wb * tau
        self.load_state_dict(state)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        torch.save(self.state_dict(), path+".torchsave")

    def load(self, path):
        self.load_state_dict(torch.load(path+".torchsave"))
        self.eval() # set batch norm layers, dropout, other stuff we don't use

class TorchDuelNetwork(TorchNetwork):
    def build_model_(self):
        self.feature = nn.Sequential(*self.build_feature_network())
        self.advantage = nn.Sequential(nn.Linear(self.hidden, self.hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden, self.actions))
        self.value = nn.Sequential(nn.Linear(self.hidden, self.hidden),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden, 1))

    def forward(self, s): 
        """
        Return tensor corresponding to Q-values when using dueling Q-networks (see exercise description)
        """
        # TODO: 4 lines missing.
        raise NotImplementedError("Implement function body")
        return value + advantage - advantage.mean()

class TorchDuelNetworkAtari(TorchNetwork):
    def build_feature_network(self):
        hidden_size = 256
        in_channels = self.env.observation_space.shape[-1]
        num_actions = self.env.action_space.n
        return (nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.Linear(7 * 7 * 64, hidden_size), # has to be adjusted for other resolutionz
                nn.Linear(hidden_size, num_actions) )

if __name__ == "__main__":
    a = 234
    import gymnasium as gym

    env = gym.make("CartPole-v0")
    Q = DQNNetwork(env, trainable=True, learning_rate=0.001)

    # self.Q = Network(env, trainable=True)  # initialize the network
    """ Assuming s has dimension [batch_dim x d] this returns a float numpy Array
    array of Q-values of [batch_dim x actions], such that qvals[i,a] = Q(s_i,a) """
    batch_size = 32 # As an example
    # Creates some dummy input
    states = [env.reset()[0] for _ in range(batch_size)]
    states.shape    # batch_size x n

    qvals = Q(states)
    qvals.shape # This is a tensor of dimension batch_size x actions
    print(qvals[0,1]) # Get Q(s_0, 1)

    Y = np.random.rand( (batch_size, 1)) # Generate target Q-values (training data)
    Q.fit(states, Y) # Train the Q-network.



    # Q = TorchNetwork()
