#!/usr/bin/env python3

import numpy as np
import rospy
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

current_observation = np.zeros(12)

def eef_pose(data):
    x_pose = data.base.tool_pose_x 
    y_pose = data.base.tool_pose_y 
    z_pose = data.base.tool_pose_z
    y_vel = data.interconnect.imu_angular_velocity_z/10
    y_twist = np.deg2rad(data.base.tool_pose_theta_y)
    z_vel = data.interconnect.imu_angular_velocity_x/10
    current_observation[0] = x_pose
    current_observation[1] = y_pose
    current_observation[2] = z_pose
    current_observation[3] = y_vel
    current_observation[4] = y_twist
    current_observation[5] = z_vel


def pen_pose(data):
    pen_pose = data.linear
    pen_orientation = data.angular
    current_observation[6] = pen_pose.x
    current_observation[7] = pen_pose.y
    current_observation[8] = pen_pose.z
    current_observation[9] = pen_orientation.x
    current_observation[10] = pen_orientation.y
    current_observation[11] = pen_orientation.z

def observation_publisher():
    pub = rospy.Publisher("rl_observation", ObsMessage, queue_size=1)
    rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, callback=eef_pose)
    rospy.Subscriber("/pen_transform_pose", Twist, callback=pen_pose)
    rospy.init_node("observation_pub", anonymous=True)
    rate = rospy.Rate(1000)

    while not rospy.is_shutdown():
        pub.publish(ObsMessage(current_observation.tolist()))
        rate.sleep()


if __name__ == '__main__':
    try:
        print("publishing observations")
        observation_publisher()
    except rospy.ROSInterruptException:
        pass