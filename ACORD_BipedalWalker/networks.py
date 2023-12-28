"""
The SAC networks mainly come from a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = F.relu(self.fc2(action_value))

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True, with_Dist=False, mask = None, with_base_action=False):
        if mask is None:
            if with_Dist:
                if not with_base_action:
                    mu, sigma = self.forward(state)
                    probabilities = Normal(mu, sigma)

                    if reparameterize:
                        actions = probabilities.rsample()
                    else:
                        actions = probabilities.sample()

                    action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
                    log_probs = probabilities.log_prob(actions)
                    log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                    log_probs = log_probs.sum(1, keepdim=True)
                    return action, log_probs, probabilities
                else:

                    mu, sigma = self.forward(state)
                    probabilities = Normal(mu, sigma)

                    if reparameterize:
                        actions = probabilities.rsample()
                    else:
                        actions = probabilities.sample()

                    action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
                    log_probs = probabilities.log_prob(actions)
                    log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                    log_probs = log_probs.sum(1, keepdim=True)
                    return action, log_probs, probabilities, actions
            else:
                state=torch.nan_to_num(state)
                mu, sigma = self.forward(state)
                try:
                    probabilities = Normal(mu, sigma)
                except Exception as e:
                    print(state)
                    print(mu, sigma)
                    raise
                if reparameterize:
                    actions = probabilities.rsample()
                else:
                    actions = probabilities.sample()

                action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
                log_probs = probabilities.log_prob(actions)
                log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                log_probs = log_probs.sum(1, keepdim=True)
                return action, log_probs
        else:
            if with_Dist:
                if not with_base_action:
                    mu, sigma = self.forward(state)
                    probabilities = Normal(mu, sigma)

                    if reparameterize:
                        actions = probabilities.rsample()
                    else:
                        actions = probabilities.sample()

                    action = torch.tanh(actions)*torch.tensor(mask).to(self.device)
                    log_probs = probabilities.log_prob(actions)
                    log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                    log_probs = log_probs.sum(1, keepdim=True)
                    return action, log_probs, probabilities
                else:

                    mu, sigma = self.forward(state)
                    probabilities = Normal(mu, sigma)

                    if reparameterize:
                        actions = probabilities.rsample()
                    else:
                        actions = probabilities.sample()

                    action = torch.tanh(actions)*torch.tensor(mask).to(self.device)
                    log_probs = probabilities.log_prob(actions)
                    log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                    log_probs = log_probs.sum(1, keepdim=True)
                    return action, log_probs, probabilities, actions
            else:
                mu, sigma = self.forward(state)
                probabilities = Normal(mu, sigma)

                if reparameterize:
                    actions = probabilities.rsample()
                else:
                    actions = probabilities.sample()

                action2 = torch.tanh(actions)*torch.tensor(mask).to(self.device)
                action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
                log_probs = probabilities.log_prob(actions)
                log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
                log_probs = log_probs.sum(1, keepdim=True)
                return action, log_probs, action2

    def get_dist(self, state, with_grad = False):
        if with_grad == False:
            with torch.no_grad():
                mu, sigma = self.forward(state)
        else:
            mu, sigma = self.forward(state)
        return mu, sigma

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256,
            fc2_dims=256, prediction_dims=1, name='discriminator', chkpt_dir='tmp/sac'):
        super(DiscriminatorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.prediction_dims = prediction_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.prediction_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.prediction_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.tanh(self.fc2(prob))*5

        mu = torch.sigmoid(self.mu(prob))
        sigma = torch.sigmoid(self.sigma(prob))

        sigma = torch.clamp(sigma, min=.1, max=1)

        return mu, sigma

    def predict(self, state, reparameterize=True, requires_grad=True):
        if requires_grad:
            mu, sigma = self.forward(state)
            probabilities = Normal(mu, sigma)
            if reparameterize:
                predictions = probabilities.rsample()
            else:
                predictions = probabilities.sample()

            prediciton = predictions.to(self.device)
            prediciton = torch.clamp(prediciton, .0001, .9999)
            log_probs = probabilities.log_prob(predictions)
            log_probs -= torch.log(1-prediciton.pow(2)+self.reparam_noise)

            return prediciton, log_probs, probabilities
        else:
            with torch.no_grad():
                mu, sigma = self.forward(state)
                probabilities = Normal(mu, sigma)

                if reparameterize:
                    predictions = probabilities.rsample()
                else:
                    predictions = probabilities.sample()

                prediciton = predictions.to(self.device)
                prediciton = torch.clamp(prediciton, .0001, .9999)
                log_probs = probabilities.log_prob(predictions)
                log_probs -= torch.log(1-prediciton.pow(2)+self.reparam_noise)
                #log_probs = log_probs.sum(0, keepdim=True)

                return prediciton, log_probs, probabilities
