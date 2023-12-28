#!/usr/bin/env python3

import numpy as np
import rospy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from kortex_driver.srv import *
from kortex_driver.msg import *
from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
import tf2_geometry_msgs #import the packages first
import PyKDL
import geometry_msgs.msg
from math import pi


if __name__ == '__main__':
    """
    Code is largely refactored from: 
    https://gist.github.com/ojura/f9aeab1c086c8a83694af403b9d8eebe
    """
    rospy.init_node('tf_echo_pen')
    tfbuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfbuffer)
    rate = rospy.Rate(1000.0)
    pen_pos_pub = rospy.Publisher('/pen_transform_pose', Twist, queue_size=10)
    print("publishing pen pose")
    while not rospy.is_shutdown():
        try:
            trans = tfbuffer.lookup_transform('base_link', 'pen1', rospy.Time())
            #print(trans)
            trans = trans.transform
            rot = PyKDL.Rotation.Quaternion(* [ eval('trans.rotation.'+c) for c in 'xyzw'] )
            #print(' '.join( [ str(eval('trans.rotation.'+c)) for c in 'xyzw'] ))
            ypr = [ i  / pi * 180 for i in rot.GetEulerZYX() ]
            rad = np.deg2rad(ypr)
            
            twist_msg = Twist()
            twist_msg.linear.x = trans.translation.x
            twist_msg.linear.y = trans.translation.y
            twist_msg.linear.z = trans.translation.z
            twist_msg.angular.x = rad[0]
            twist_msg.angular.y = rad[1]
            twist_msg.angular.z = rad[2]
            pen_pos_pub.publish(twist_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            #print ("Fail", e)
            pass


        rate.sleep()