#!/usr/bin/env python3

import imp
import numpy as np
import time
import pickle, torch
from drawing_env_acord import SimDrawing
import rospy 
from sim_drawing.msg import LossInfoDoubleDisc
from custom_arm_reaching.msg import *
from custom_arm_reaching.srv import GetActionFromObs, GetActionFromObsResponse

from tensorboardX import SummaryWriter

writer = SummaryWriter()
env_interacts = 0

# Function that will qeuery the model for an action
def get_action(observation):
    rospy.wait_for_service('get_action')
    try:
        service = rospy.ServiceProxy('get_action', GetActionFromObs)
        resp1 = service(observation)
        #print("Obs, ", observation, "action", resp1.action)
        return resp1.action
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def write_loss(data):
    global writer
    act_loss = data.act_loss
    disc1_loss = data.disc1_loss
    disc2_loss = data.disc2_loss
    disc1_log_probs = data.disc1_log_probs
    disc2_log_probs = data.disc2_log_probs
    disc1_crit = data.disc_crit

    writer.add_scalar("Loss/act_new",act_loss,env_interacts)
    writer.add_scalar("Loss/disc1_loss_new", disc1_loss,env_interacts)
    writer.add_scalar("Loss/disc2_loss_new", disc2_loss,env_interacts)
    writer.add_scalar("Loss/disc1_log_pob_new", disc1_log_probs,env_interacts)
    writer.add_scalar("Loss/disc2_log_pob_new", disc2_log_probs,env_interacts)
    writer.add_scalar("Loss/disc1_crit_new", disc1_crit,env_interacts)
   

def main_loop():
    rospy.init_node('main_learner', anonymous=True)
    transition_pub = rospy.Publisher('rl_transition', RLTransitionPlusReward, queue_size=10, latch=True)
    save_pub = rospy.Publisher('save_agent', SaveAgentRequest, queue_size=1, latch=True)
    learning_pub = rospy.Publisher('rl_learn', LearnRequest, queue_size=1, latch=True)
    rospy.Subscriber("loss_info", LossInfoDoubleDisc, write_loss)

    rate = rospy.Rate(100) # 100hz

    env = SimDrawing(sim=True, success_threshold=.005)

    n_games = 100001
    rewards = []

    best_score = -500
    score_history = []
    load_checkpoint = False


    render = False

    old_reward = 0

    # save_req = SaveAgentRequest()
    # save_req.save = True
    # save_req.file_name = f"agents/zandrot_MultiStep_withActions_ep487.p"
    # save_pub.publish(save_req)
    episode_length = 1000
    for i in range(601, n_games):
        observation = env.reset()
        done = False
        score = 0
        limit_factor1 = np.random.uniform(low=0, high=1)
        limit_factor2 = np.random.uniform(low=0, high=1)
        episode_start_time = time.time()

        reward = -100
        old_reward = -1000
        episode_interacts = 0
        observation = np.concatenate((observation, np.zeros(4)))
        print(len(observation))
        observation = np.concatenate((observation, [limit_factor1, limit_factor2]))
        print(len(observation))
        while not done and episode_interacts < episode_length:
            global env_interacts
            env_interacts+=1
            episode_interacts+=1

            action = get_action(np.array(observation.tolist()))

            observation_, reward, done = env.step(action, velocity_control=True)
            
            observation_ = np.concatenate((observation_, action))
            observation_ = np.concatenate((observation_, [limit_factor1, limit_factor2]))
            score += reward
            #print(reward)
            #if reward > 10:
            #    print("big reward")
            temp_rew = reward
            if (reward <= old_reward and old_reward<0) or reward < -15:
               reward = min(-15, reward)
               #print("BAD REWARD")
            else:
               reward = 0
            #print(reward)

            transition = RLTransitionPlusReward()
            transition.old_observation = list(observation)
            transition.action = action
            transition.observation = list(observation_)
            transition.reward = [reward]
            transition.done = [done]
            transition_pub.publish(transition)
            
            old_reward = temp_rew


            if env_interacts % 250 == 0:
                try:
                    learn_req = LearnRequest()
                    learn_req.update_disc = True
                    learn_req.update_params = True
                    learning_pub.publish(learn_req)
                except Exception as e:
                    print(e)
                    raise
            else:        
                try:
                    learn_req = LearnRequest()
                    learn_req.update_disc = False
                    learn_req.update_params = True
                    learning_pub.publish(learn_req)
                    
                except Exception as e:
                    print(e)
                    raise

            observation = observation_

            if episode_interacts % int(episode_length/2) == 0:
                print("switching")
                limit_factor1 = np.random.uniform(low=0, high=1)
                limit_factor2 = np.random.uniform(low=0, high=1)
                observation[-2] = limit_factor1
                observation[-1] = limit_factor2

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score


        rewards.append(score)

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f, ' % avg_score, "limit_factor %.2f" % limit_factor1,  f"time steps: {env_interacts}, 50 update")

        if i % 100 == 0 and i > 10:
            save_req = SaveAgentRequest()
            save_req.save = True
            save_req.file_name = f"acord_heightAndRotation_ep{i}.p"
            save_pub.publish(save_req)


if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass