#!/usr/bin/env python3

from turtle import speed
import numpy as np
import rospy 
import time
from gen3_testing.gen3_movement_utils import Arm
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import os

# z_min = -.023
# z_max = .1
# x_min = .6
# x_max = .8
# y_min = -.27
# y_max = .25
# wrist_rotate_min = -.55
# wrist_rotate_max = .55

start_pose = [-0.14078960892369263, 1.2184466656139084, 3.0414530029128515, -2.0302097243289774, -0.17623628303264738, 1.6307301013368574, 1.635055074947146]
class SimDrawing():
    def __init__(self, max_action=.1, min_action=-.1, n_actions=1, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.1, home_arm=True, with_pixels=False, max_vel=.12, 
        duration_timeout=1, speed=1, sim=True, use_goals_file=None, z_min = -.023, z_max = .1, x_min = .6, x_max = .8, \
            y_min = -.27, y_max = .25, wrist_rotate_min = -.55, wrist_rotate_max = .55, \
                fixed_policy_vel=.01, fixed_policy_deadzone=.001, default_z=0.10229, default_rot=0.0):
        
        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.n_actions = n_actions
        self.reset_pose = reset_pose
        self.episode_time = episode_time
        self.stack_size = stack_size
        self.sparse_rewards = sparse_rewards
        self.success_threshhold = success_threshold
        self.home_arm = home_arm
        self.with_pixels = with_pixels
        self.max_vel = max_vel
        self.duration_timeout = duration_timeout
        self.sim = sim
        self.arm = Arm()
        self.goal = [0.0,0.0]
        self.speed = speed
        self.z_min = z_min
        self.z_max = z_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.wrist_rotate_min = wrist_rotate_min
        self.wrist_rotate_max = wrist_rotate_max
        self.fixed_policy_vel = fixed_policy_vel
        self.fixed_policy_deadzone = fixed_policy_deadzone
        self.default_z = default_z
        self.default_rot = default_rot
        
        self.arm.goto_cartesian_relative_sim([0, 0,.04,0,0,0], duration=self.duration_timeout, speed=self.speed)
        
        if use_goals_file is not None:
            self.goals = np.genfromtxt(use_goals_file, delimiter=',')
            self.use_goals = True
        else:
            self.use_goals = False
        # self.arm.close_gripper()
        self.goal_index = 0
        self.stop_publisher = rospy.Publisher(
                "/my_gen3/in/stop", std_msgs.msg.Empty, queue_size=1, latch=True)
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        stop_publisher = rospy.Publisher(
                "/my_gen3/in/stop", std_msgs.msg.Empty, queue_size=1, latch=True)
        stop_publisher.publish(std_msgs.msg.Empty())
        
    def get_obs(self):
        temp = np.array(rospy.wait_for_message("/rl_observation", ObsMessage).obs)
        temp = np.concatenate((temp, self.goal))
        return temp

    def reset(self, just_home=False):
        if just_home:
            self.arm.home_arm()
            return
        else: 
            global goal
            time.sleep(.5)
            current_eef_pose = self.arm.get_eef_pose(quaternion=False)
            if self.use_goals:
                self.goal_index = 0
                self.goal = self.goals[self.goal_index]
                self.goal_index += 1
            else:
                self.goal[0] = np.random.uniform(self.x_min, self.x_max)
                self.goal[1] = np.random.uniform(self.y_min, self.y_max)

            self.arm.home_arm()
            if self.sim:
                self.arm.goto_joint_pose_sim(start_pose)
            else:
                self.arm.goto_joint_pose_sim(start_pose)
            
            time.sleep(2)
            if not self.sim:
                self.arm.goto_cartesian_pose([self.goal[0], self.goal[1], current_eef_pose[2], current_eef_pose[3], \
                    current_eef_pose[4],current_eef_pose[5]], radians=True)
                time.sleep(.5)
                self.arm.goto_cartesian_pose([self.goal[0], self.goal[1], self.default_z, current_eef_pose[3], \
                        self.default_rot,current_eef_pose[5]], radians=True)
            else:
                self.arm.goto_cartesian_pose_sim([self.goal[0], self.goal[1], current_eef_pose[2], current_eef_pose[3], \
                    current_eef_pose[4],current_eef_pose[5]], radians=True)
                time.sleep(.5)
                self.arm.goto_cartesian_pose_sim([self.goal[0], self.goal[1], self.default_z, current_eef_pose[3], \
                        self.default_rot,current_eef_pose[5]], radians=True)
            
            rospy.sleep(1)
            return self.get_obs()

    def step(self, action, velocity_control=False, used_fixed_policy=False):
        #print(action)
        action = np.clip(np.array(action)*self.max_action, self.min_action, self.max_action)
        action_xy = [0,0]
        obs= self.get_obs()
        if obs[6] < self.goal[0]:
            if np.abs(obs[6] - self.goal[0]) < self.fixed_policy_deadzone:
                action_xy[0] = 0
            else:
                action_xy[0] = self.fixed_policy_vel
        elif obs[6] > self.goal[0]:
            if np.abs(obs[6] - self.goal[0]) < self.fixed_policy_deadzone:
                action_xy[0] = 0
            else:    
                action_xy[0] = -self.fixed_policy_vel
        if obs[7] < self.goal[1]:
            if np.abs(obs[7] - self.goal[1]) < self.fixed_policy_deadzone:
                action_xy[1] = 0
            else:
                action_xy[1] = self.fixed_policy_vel
        elif obs[7] > self.goal[1]:
            if np.abs(obs[7] - self.goal[1]) < self.fixed_policy_deadzone:
                action_xy[1] = 0
            else: 
                action_xy[1] = -self.fixed_policy_vel
        
        # check pen z and wrist rotate limits
        if obs[8] < self.z_min + .005:
            action[2] = 0
        elif obs[8] > self.z_max - .005:
            action[2] = 0
        if obs[10] < self.wrist_rotate_min + .005:
            action[3] = 0
        elif obs[10] > self.wrist_rotate_max - .005:
            action[3] = 0

        if used_fixed_policy:
            if self.sim:
                self.arm.goto_cartesian_relative_sim(.5*np.array([action_xy[0],action_xy[1],action[2],0.0,action[3],0.0]), duration=self.duration_timeout, speed=self.speed)
                rospy.sleep(.25)
                empty_message = std_msgs.msg.Empty()
                self.stop_publisher.publish(empty_message)

            else:
                if not velocity_control:
                    self.arm.goto_cartesian_pose(np.array([action_xy[0],action_xy[1],action[2],0.0,action[3],0.0]), relative=True, radians=True, wait_for_end=False)
                else:             
                    self.arm.cartesian_velocity_command(.5*np.array([action_xy[0],action_xy[1],action[2],0,0,2*-action[3]]), duration=.05, radians=True)
        else:
            if self.sim:
                self.arm.goto_cartesian_relative_sim(.1*np.array([action[0],action[1],action[2],0.0,action[3],0.0]), duration=self.duration_timeout, speed=self.speed)
                rospy.sleep(.25)
                empty_message = std_msgs.msg.Empty()
                self.stop_publisher.publish(empty_message)
            else:
                if not velocity_control:
                    self.arm.goto_cartesian_pose(np.array([action[0],action[1],action[2],0.0,action[3],0.0]), relative=True, radians=True, wait_for_end=False)
                else:             
                    self.arm.cartesian_velocity_command(.1*np.array([action[0],action[1],action[2],0,0,2*-action[3]]), duration=.05, radians=True)
        obs = self.get_obs()
        reward = 0
        done = False
        reward = np.abs((obs[6]-self.goal[0])) + np.abs((obs[7]-self.goal[1])) 
        if reward <= self.success_threshhold:
            reward = -reward
            if self.sparse_rewards:
                reward = 0
            if self.use_goals:
                if self.goal_index >= len(self.goals):
                    done = True
                else:
                    self.goal = self.goals[self.goal_index]
                    self.goal_index += 1
            else:
                self.goal[0] = np.random.normal(self.goal[0], .03)
                self.goal[1] = np.random.normal(self.goal[1], .03)
                self.goal[0] = np.clip(self.goal[0], self.x_min, self.x_max)
                self.goal[1] = np.clip(self.goal[1], self.y_min, self.y_max)
                obs = self.get_obs()
           
        else:
            reward = -reward
            if self.sparse_rewards:
                reward = -1
            done = False
        
        if obs[6] < self.x_min-.1 or obs[6] > self.x_max+.1 or obs[7] < self.y_min-.1 or obs[7] > self.y_max+.1:
           print("eef out of bounds")
           self.stop_publisher.publish()
           reward = -100
           done = True
        
        if obs[8] < self.z_min or obs[8] > self.z_max:
           reward = -60
           print('bad z')
           self.stop_publisher.publish()
           done = True
                
        if obs[10] < self.wrist_rotate_min or obs[10] > self.wrist_rotate_max:
           print("wrist out of bounds")
           reward = -20
           self.stop_publisher.publish()
           done = True
        

        return obs, reward, done, action_xy

if __name__ == '__main__':
    try:
        rospy.init_node("custom_reacher")
        SimDrawing()
    except rospy.ROSInterruptException:
        pass

        