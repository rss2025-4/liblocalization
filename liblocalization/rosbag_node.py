from typing import Callable

import tf2_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from odom_transformer.transformer import Transformer
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    Node,
    PoseWithCovarianceStamped,
    TransformStamped,
    rclpy,
)
from visualization_msgs.msg import Marker

from liblocalization import deterministic_motion_tracker
from liblocalization.api import LocalizationBase, localization_params
from liblocalization.controllers.particles import particles_model, particles_params


class RosbagNode(Node):
    def __init__(
        self, controller_init: Callable[[localization_params], LocalizationBase]
    ):
        super().__init__("RosbagNode")

        self.controller_init = controller_init
        self.controller = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid, "map", self.map_callback, 1
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/vesc/odom", self.odom_callback, 1
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 1
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1
        )

        self.visualization_pub = self.create_publisher(Marker, "/visualization", 10)

        self.finished_init = False
        self.odom_transformer = None

    def map_callback(self, map_msg: OccupancyGrid):
        self.controller = self.controller_init(
            localization_params(
                n_laser_points=1081,
                map=map_msg,
                marker_callback=self.marker_callback,
                ground_truth_callback=self.ground_truth_callback,
            )
        )
        assert isinstance(self.controller, LocalizationBase)

    def get_controller(self) -> LocalizationBase | None:
        if self.controller is None:
            return None

        if not self.finished_init:
            try:
                self.odom_transformer = Transformer(
                    self.tfBuffer.lookup_transform(
                        "laser", "base_link", Time()
                    ).transform
                )
            except Exception as e:
                print("failed to get transform:", e)
                return None

            self.finished_init = True

        return self.controller

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        if controller := self.get_controller():
            assert self.odom_transformer is not None
            pos_laser = self.odom_transformer.transform_pose(msg.pose)

            t = TransformStamped()
            t.header = msg.header
            t.child_frame_id = "laser"

            pos = pos_laser.pose.position
            t.transform.translation.x = pos.x
            t.transform.translation.y = pos.y
            t.transform.translation.z = pos.z

            t.transform.rotation = pos_laser.pose.orientation

            controller.set_pose(t)

    def odom_callback(self, msg: Odometry):
        if controller := self.get_controller():
            assert self.odom_transformer is not None

            assert msg.child_frame_id == "base_link"

            assert msg.twist.twist.linear.y == 0.0
            msg.twist.twist.linear.x = -msg.twist.twist.linear.x
            msg.twist.twist.angular.z = -msg.twist.twist.angular.z

            odom = Odometry(header=msg.header, child_frame_id="laser")
            odom.pose = self.odom_transformer.transform_pose(msg.pose)
            odom.twist = self.odom_transformer.transform_twist(msg.twist)

            # TODO: real messages have no timestamps
            msg.header.stamp = self.get_clock().now().to_msg()

            controller.odom_callback(odom)

            # print("pose:", controller.get_pose())
            # print("particles:", controller.get_particles())

    def lidar_callback(self, msg: LaserScan):
        if controller := self.get_controller():

            # TODO: real messages have no timestamps
            msg.header.stamp = self.get_clock().now().to_msg()

            controller.lidar_callback(msg)

    def marker_callback(self, marker: Marker):
        self.visualization_pub.publish(marker)

    def ground_truth_callback(self) -> TransformStamped | None:
        if controller := self.get_controller():
            pose = controller.get_pose()
            conf = controller.get_confidence()
            if conf > -3.5:
                return pose


def examplemain():
    """examle (deterministic_motion_tracker)"""
    rclpy.init()
    rclpy.spin(RosbagNode(deterministic_motion_tracker))
    assert False, "unreachable"


def examplemain2():
    """examle (particles_model)"""
    rclpy.init()
    rclpy.spin(RosbagNode(particles_model(particles_params(n_particles=1000))))
    assert False, "unreachable"
