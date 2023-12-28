#!/usr/bin/env python3

from glob import glob
import numpy as np
import rospy 
import pickle, gc
from acord import Agent
from custom_arm_reaching.msg import *
from sim_drawing.msg import LossInfoDoubleDisc
from custom_arm_reaching.srv import GetActionFromObs, GetActionFromObsResponse
from buffer import ReplayBuffer

global agent
agent = Agent(alpha=.0003, beta=.0003, disc_lr=.0001, input_dims=(20,), batch_size=256, disc_layer1_size=256, disc_layer2_size=256,
            tau=.02, max_size=100000, layer1_size=400, layer2_size=300, n_actions=4, reward_scale=15, 
            disc_input_dims=(1,), predict_dims=1)

def store_transition(data):
    old_observation = np.array(data.old_observation)
    action = np.array(data.action)
    reward = np.array(data.reward)
    observation = np.array(data.observation)
    done = np.array(data.done)
    global agent
    agent.remember(old_observation, action, reward, observation, done)

def learn(data):
    update_params = data.update_params
    update_disc = data.update_disc
    global agent
    gc.collect()
    if update_params and update_disc:
        try:
            act_loss, disc1_loss, disc2_loss, disc1_log_probs, disc2_log_probs,\
                        disc_crit = agent.learn(update_params=True, update_disc=True)
            if not act_loss == 0 and not disc1_loss == 0:
                loss_pub = rospy.Publisher("loss_info", LossInfoDoubleDisc, queue_size=1, latch=True)
                losses = LossInfoDoubleDisc()
                losses.act_loss = act_loss
                losses.disc1_loss = disc1_loss
                losses.disc2_loss = disc2_loss
                losses.disc1_log_probs = disc1_log_probs
                losses.disc2_log_probs = disc2_log_probs
                losses.disc_crit = disc_crit
                loss_pub.publish(losses)
                agent.garbage_collect()
        except Exception as e:
            print(e)
            pass
    elif update_params and not update_disc:
        agent.learn(update_params=True)
    elif update_disc and not update_params:
        agent.learn(update_params=True)
    else:
        try:
            act_loss, disc1_loss = agent.learn(update_params=False, update_disc=False)
        except Exception as e:
            pass

def action_callback(observation):
    observation = observation.observation
    action = agent.choose_action(observation)
    return GetActionFromObsResponse(action)

def save_callback(data):
    save = data.save
    global agent
    if save:
        pickle.dump(agent, open(data.file_name, "wb" ) )
    else:
        return

def learning_listener():
    rospy.init_node('learner', anonymous=True)

    action_service = rospy.Service('get_action', GetActionFromObs, action_callback)
    rospy.Subscriber("/rl_transition", RLTransitionPlusReward, store_transition)
    rospy.Subscriber("/save_agent", SaveAgentRequest, save_callback)
    rospy.Subscriber("/rl_learn", LearnRequest, learn, queue_size=1, buff_size=2)
    print("Ready to recieve actions")
    # learn()

    rospy.spin()

if __name__ == '__main__':
    learning_listener()
