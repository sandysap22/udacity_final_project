#!/usr/bin/env python
import numpy as nump
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
Carla_Update_Frequency = 50 # 50 Hz: Update sent 50 times a second.
maximum_deceleration = 0.5 #Maximum Deceleration


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        #rospy.Subscriber('/traffic_waypoint', PoseStamped, self.pose_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.waypoints_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_lane = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stop_line_waypoint_index = -1
        
        self. loop()
    
    def loop(self):
        frequency = rospy.Rate(Carla_Update_Frequency)
        while not rospy.is_shutdown():
            #if self.pose and self.base_waypoints:
                #Get Closest Waypoint:
                #closest_waypoint_index = self.get_closest_point_index()
                #self.publish_waypoints(closest_waypoint_index)
            if (self.pose and self.base_lane):
                self.publish_waypoints()
                
            frequency.sleep()
    
    def get_closest_point_index(self):
        x_coord = self.pose.pose.position.x
        y_coord = self.pose.pose.position.y
        
        closest_point_index = self.waypoint_tree.query([x_coord, y_coord], 1)[1]
        
        # Check if the Closest Waypoint is Ahead, or Behind Carla:
        closest_coordinate = self.waypoints_2d[closest_point_index]
        previous_coordinate = self.waypoints_2d[closest_point_index - 1]
        
        # Equation for Hyper-Plane through the Closest Co-Ordinates:
        closest_point_vector = nump.array(closest_coordinate)
        previous_point_vector = nump.array(previous_coordinate)
        position_vector = nump.array([x_coord, y_coord])
        
        value = nump.dot(closest_point_vector - previous_point_vector, position_vector - closest_point_vector)
        
        if value > 0:
            closest_point_index = (closest_point_index + 1) % len(self.waypoints_2d)
        return closest_point_index
    
    #def publish_waypoints(self, closest_point_index):
    def publish_waypoints(self):
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_point_index: closest_point_index + LOOKAHEAD_WPS]
        #self.final_waypoints_pub.publish(lane)
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
    
    def generate_lane(self):
        lane = Lane()
        
        closest_waypoint_index = self.get_closest_point_index()
        farthest_waypoint_index = closest_waypoint_index + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_waypoint_index: farthest_waypoint_index]
        
        if ((self.stop_line_waypoint_index == -1) or (self.stop_line_waypoint_index >= farthest_waypoint_index)):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_waypoint_index)

        return lane

    def decelerate_waypoints(self, waypoints, closest_waypoint_index):
        placeholder = []

        for i, waypoint in enumerate(waypoints):
            way_point = Waypoint()
            way_point.pose = waypoint.pose

            stop_line_index = max(self.stop_line_waypoint_index - closest_waypoint_index - 2, 0) # 2 waypoints back from the Stop-Line
            distance = self.distance(waypoints, i, stop_line_index)
            
            #velocity = math.sqrt(2 * maximum_deceleration * distance)
            velocity = math.sqrt(2 * maximum_deceleration * distance)
            
            if (velocity < 1.0):
                velocity = 0
            
            way_point.twist.twist.linear.x = min(velocity, waypoint.twist.twist.linear.x)
            placeholder.append(way_point)
            
        return placeholder
        #rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #self.base_waypoints = waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stop_line_waypoint_index = msg.data
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
