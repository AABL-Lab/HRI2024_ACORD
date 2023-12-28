"""
The SAC code mainly follows a Soft-Actor Critic YouTube tutorial (with modifications) found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil
"""


import gc
import os
from re import A
import torch
from torch._C import device 
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork, DiscriminatorNetwork
import copy

class Agent():
    def __init__(self, alpha=0.001, beta=0.001, disc_lr=.0001, input_dims=[8], 
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, disc_layer1_size=256, disc_layer2_size=256, disc_batch_size=128, batch_size=256, reward_scale=2, auto_entropy=False, 
            entr_lr=None, reparam_noise=1e-6, disc_input_dims=[8], predict_dims=1, max_action=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.disc_memory = ReplayBuffer(1000000, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = max_action
        self.env = env
        self.auto_entropy = True
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='actor', max_action=self.max_action)
        self.critic_1 = CriticNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_2')

        self.disc_critic_1 = CriticNetwork(beta, input_dims, fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, n_actions=n_actions,
                    name='disc_critic_1')
        self.disc_critic_2 = CriticNetwork(beta, input_dims,fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, n_actions=n_actions,
                    name='disc_critic_2')

        self.limit_factor_dist = torch.distributions.Uniform(low=torch.tensor(0.0).to('cuda:0' if torch.cuda.is_available() else 'cpu'),  high=torch.tensor(1.0).to('cuda:0' if torch.cuda.is_available() else 'cpu'))

        #self.discriminator = DiscriminatorNetwork(lr=disc_lr, input_dims=[input_dims[0]-1], fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, prediction_dims=predict_dims)
        self.discriminator1 = DiscriminatorNetwork(lr=disc_lr, input_dims=disc_input_dims, fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, prediction_dims=predict_dims)
        self.discriminator2 = DiscriminatorNetwork(lr=disc_lr, input_dims=disc_input_dims, fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, prediction_dims=predict_dims)
        self.value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.target_entropy = -np.prod((self.n_actions,)).astype(np.float32)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        if entr_lr is None:
            self.entr_lr = alpha
        else:
            self.entr_lr = entr_lr
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha)
        self.entropy = 0


    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        self.disc_memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
    
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
    
    def garbage_collect(self):
        gc.collect()
    def learn(self, update_params=True, update_disc=False):
        if(self.auto_entropy):
            actor_loss = None
            if self.memory.mem_cntr < self.batch_size:
                return None, None

            state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)

            reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
            done = torch.tensor(done).to(self.actor.device)
            state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

            value = self.value(state).view(-1)
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0

            limit_factor = torch.clone(state)
            limit_factor = limit_factor[:, -2:]
            limit_factor1 = limit_factor[:,0]
            limit_factor2 = limit_factor[:,1]
            limit_factor1 = limit_factor1[:,None]
            limit_factor2 = limit_factor2[:,None]

            disc_state = torch.clone(state)
            # zpos and y-angle
            disc1_state = disc_state[:, 8:9]
            disc2_state = disc_state[:, 10:11]

            disc_new_state = torch.clone(state_)
            # disc1_new_state = disc_new_state[:, 8:9]
            # disc2_new_state = disc_new_state[:, 10:11]


            actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            #env_critic_value = torch.min(q1_new_policy, q2_new_policy) 
            disc_q1_new_policy = self.disc_critic_1.forward(state, actions)
            disc_q2_new_policy = self.disc_critic_2.forward(state, actions)
            disc_critic_value = torch.min(disc_q1_new_policy, disc_q2_new_policy) 
            #critic_value = torch.where(limit_factor>.5, ((2*limit_factor-1)*env_critic_value+(1-(2*limit_factor-1))*disc_critic_value), ((1-2*limit_factor)*env_critic_value+(2*limit_factor)*disc_critic_value))
            #critic_value = scaled_limit_factor*disc_critic_value + (1-scaled_limit_factor)*env_critic_value
            critic_value = disc_critic_value
            critic_value = critic_value.view(-1)

            self.value.optimizer.zero_grad()
            value_target = (-self.entropy*log_probs + critic_value)
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()
            
            actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            #env_critic_value = torch.min(q1_new_policy, q2_new_policy) 
            disc_q1_new_policy = self.disc_critic_1.forward(state, actions)
            disc_q2_new_policy = self.disc_critic_2.forward(state, actions)
            disc_critic_value = torch.min(disc_q1_new_policy, disc_q2_new_policy)
            #critic_value = torch.where(limit_factor>.5, ((2*limit_factor-1)*env_critic_value+(1-(2*limit_factor-1))*disc_critic_value), ((1-2*limit_factor)*env_critic_value+(2*limit_factor)*disc_critic_value))
            #critic_value = scaled_limit_factor*disc_critic_value + (1-scaled_limit_factor)*env_critic_value
            critic_value = disc_critic_value
            critic_value = critic_value.view(-1)

            actor_loss = (self.entropy*log_probs - critic_value) 
            #actor_loss = (log_probs - critic_value) 
            if torch.any(torch.isinf(actor_loss)) or torch.any(torch.isnan(actor_loss)):
                print(actor_loss, "actor_loss")
                print(log_probs, "log_probs")
                print(critic_value, "critic value")
            actor_loss = torch.mean(actor_loss)

            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()


            disc1_predictions, disc1_log_probs, dist1 = self.discriminator1.predict(disc1_state, requires_grad=False)
            disc1_log_probs.to('cuda:0' if torch.cuda.is_available() else 'cpu')


            value_ = torch.clone(value_).detach().to('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # log probability of the limit factor
            log_prob_of_lf1 = self.limit_factor_dist.log_prob(disc1_predictions.detach())

            log_prob_of_lf1 = torch.clone(log_prob_of_lf1).to('cuda:0' if torch.cuda.is_available() else 'cpu')

            

            if torch.any(torch.isinf(disc1_log_probs)) or torch.any(torch.isnan(disc1_log_probs)) :
                print(disc1_log_probs, "disc_log_probs")
            if torch.any(torch.isinf(log_prob_of_lf1)) or torch.any(torch.isnan(log_prob_of_lf1)):
                log_prob_of_lf1 = torch.nan_to_num(log_prob_of_lf1, posinf=0, neginf=0)

            disc2_predictions, disc2_log_probs, dist2 = self.discriminator2.predict(disc2_state, requires_grad=False)
            disc2_log_probs.to('cuda:0' if torch.cuda.is_available() else 'cpu')


            log_prob_of_lf2 = self.limit_factor_dist.log_prob(disc2_predictions.detach())

            log_prob_of_lf2 = torch.clone(log_prob_of_lf2).to('cuda:0' if torch.cuda.is_available() else 'cpu')

            

            if torch.any(torch.isinf(disc2_log_probs)) or torch.any(torch.isnan(disc2_log_probs)) :
                print(disc2_log_probs, "disc_log_probs")
            if torch.any(torch.isinf(log_prob_of_lf2)) or torch.any(torch.isnan(log_prob_of_lf2)):
                log_prob_of_lf2 = torch.nan_to_num(log_prob_of_lf2, posinf=0, neginf=0)
                #print(log_prob_of_lf, "log_prob_of_lf") 

            if torch.any(torch.isinf(value_)) or torch.any(torch.isnan(value_)):
                print(value_, "log_prob_of_lf")    


            rew=(-torch.log(torch.clamp(torch.abs(disc1_predictions-limit_factor1)**2, min=.000001, max=.99999)) + \
                -torch.log(torch.clamp(torch.abs(disc2_predictions-limit_factor2)**2, min=.000001, max=.99999)))/2
            
            if torch.any(torch.isinf(rew)) or torch.any(torch.isnan(rew)):
                print(rew, "rew")    
                print((dist1.cdf(disc1_predictions)-dist1.cdf(limit_factor))**2, "diff")
            
            rew = (rew[:,-1])*12
            rew = torch.where(reward<-14, reward*2, rew)
            
            disc_q_hat = (rew) + self.gamma*value_
            disc_q1_old_policy = self.disc_critic_1.forward(state, action).view(-1)
            disc_q2_old_policy = self.disc_critic_2.forward(state, action).view(-1)
            disc_critic_1_loss = 0.5 * F.mse_loss(disc_q1_old_policy, disc_q_hat)
            disc_critic_2_loss = 0.5 * F.mse_loss(disc_q2_old_policy, disc_q_hat)

            self.disc_critic_1.optimizer.zero_grad()
            self.disc_critic_2.optimizer.zero_grad()
            disc_critic_loss = disc_critic_1_loss + disc_critic_2_loss
            disc_critic_loss.backward()
            self.disc_critic_1.optimizer.step()
            self.disc_critic_2.optimizer.step()
            disc_loss = None
            disc1_loss = None
            disc2_loss = None
            if update_disc:
                state, action, reward, new_state, done = \
                    self.disc_memory.sample_buffer(self.batch_size)

                reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
                done = torch.tensor(done).to(self.actor.device)
                state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
                state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
                action = torch.tensor(action, dtype=torch.float).to(self.actor.device)


                limit_factor = torch.clone(state)
                limit_factor = limit_factor[:, -2:]
                limit_factor1 = limit_factor[:,0]
                limit_factor2 = limit_factor[:,1]
                limit_factor1 = limit_factor1[:,None]
                limit_factor2 = limit_factor2[:,None]

                disc_state = torch.clone(state)
                # zpos and y-angle
                disc1_state = disc_state[:, 8:9]
                disc2_state = disc_state[:, 10:11]
                
                # with torch.no_grad():
                disc1_predictions, disc1_log_probs, dist1 = self.discriminator1.predict(disc1_state, requires_grad=True)
                print("dist1", max(dist1.loc), min(dist1.loc))
                print("dist1", max(dist1.scale), min(dist1.scale))
                self.discriminator1.optimizer.zero_grad()
                disc1_loss = (F.mse_loss(disc1_predictions, limit_factor1)*10+ (1/torch.abs((min(dist1.loc)-max(dist1.loc)))))*10 
                print("mse loss1: ", F.mse_loss(disc1_predictions, limit_factor1))
                disc1_loss.backward()
                self.discriminator1.optimizer.step()

                disc2_predictions, disc2_log_probs, dist2 = self.discriminator2.predict(disc2_state, requires_grad=True)
                print("dist2", max(dist2.loc), min(dist2.loc))
                print("dist2", max(dist2.scale), min(dist2.scale))
                self.discriminator2.optimizer.zero_grad()
                disc2_loss = (F.mse_loss(disc2_predictions, limit_factor2)*10+ torch.nan_to_num((1/(torch.abs((min(dist2.loc)-max(dist2.loc)))+.0001))))*10 
                print("mse loss2: ", F.mse_loss(disc2_predictions, limit_factor2))
                disc2_loss.backward()
                self.discriminator2.optimizer.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.entropy = self.log_alpha.exp()

            if update_params:
                self.update_network_parameters()
            
            if disc1_loss is not None and disc2_loss is not None:
                return np.mean(actor_loss.item()), np.mean(disc1_loss.item()), np.mean(disc2_loss.item()), torch.mean(disc1_log_probs).item(),\
                torch.mean(disc2_log_probs).item(), torch.mean(critic_value).item()
            else:
                return actor_loss, disc_loss
        