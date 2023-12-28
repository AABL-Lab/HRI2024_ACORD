from ast import expr_context
import gym
import numpy as np
from ACORD import Agent
import time
import pickle

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    print(env.observation_space.shape)
    in_dims = (26,)
    agent = Agent(alpha=.0003, beta=.0003, disc_lr=.0001, input_dims=in_dims, env=env, batch_size=256, disc_layer1_size=256, disc_layer2_size=256,
            tau=.02, max_size=100000, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], reward_scale=15, disc_input_dims=(1,), predict_dims=1)
    agent.actor.max_action=1
    n_games = 5001
    rewards = []


    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    render = False
    env_interacts = 0

    writer = SummaryWriter()
    action = env.action_space.sample()
    for i in range(0, n_games):
        observation = env.reset()

        done = False
        score = 0
        limit_factor = np.random.uniform(low=0, high=1)
        limit_factor2 = np.random.uniform(low=0, high=1)
        observation = np.concatenate((observation[0], [limit_factor, limit_factor2]))
        while not done:
            env_interacts+=1

            action = agent.choose_action(observation)

            info = env.step(action)
            observation_ = info[0]
            reward = info[1]
            done = info[2]

            observation_ = np.concatenate((observation_, [limit_factor, limit_factor2]))
            score += reward
            if observation_[2] < 0 or reward < -70:
                if reward < -70:
                    reward = -200
                else:
                    reward = -5

            agent.remember(observation, action, reward, observation_, done)

            if env_interacts % 1000 == 0:
                try:
                    act_loss, disc1_loss, disc2_loss, disc1_log_probs, disc2_log_probs,\
                        disc_crit = agent.learn(update_params=True, update_disc=True)
                    if act_loss is not None and disc1_loss is not None:
                        writer.add_scalar("Loss/act_new", np.mean(act_loss.item()),env_interacts)
                        writer.add_scalar("Loss/disc1_loss_new", np.mean(disc1_loss.item()),env_interacts)
                        writer.add_scalar("Loss/disc2_loss_new", np.mean(disc2_loss.item()),env_interacts)
                        writer.add_scalar("Loss/disc_crit_new", disc_crit,env_interacts)
                        writer.add_scalar("Loss/disc1_log_pob_new", disc1_log_probs,env_interacts)
                        writer.add_scalar("Loss/disc2_log_pob_new", disc2_log_probs,env_interacts)

                except:
                    pass
            else:
                try:
                    act_loss, _ = agent.learn()
                except Exception as e:
                    print(e)
                    raise

            if env_interacts % 500 == 0:
                limit_factor = np.random.uniform(low=0, high=1)
                limit_factor2 = np.random.uniform(low=0, high=1)
                observation_[-2] = limit_factor
                observation_[-1] = limit_factor2
                observation = observation_
            else:
                observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        rewards.append(score)

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f, ' % avg_score, "limit_factor %.2f" % limit_factor,  ", 0003, 0001, 378 speedAndHullDoubleDisc")

        if i % 250 == 0 and i > 990:
            pickle.dump(agent, open( f"ACORD_speedAndHull_ep{i}.p", "wb" ) )