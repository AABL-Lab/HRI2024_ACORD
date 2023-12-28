#!/usr/bin/env python3

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('fixed_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    print("publishing transform") 
    while not rospy.is_shutdown():
        br.sendTransform((0.0, -0.06, 0.025),
                         (0.0, 0.0, 0.0, 1.0),
                         rospy.Time.now(),
                         "pen1",
                         "tool_frame")
        rate.sleep()