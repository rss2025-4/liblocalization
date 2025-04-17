from dataclasses import dataclass
from typing import Callable

import tf2_ros
from geometry_msgs.msg import PoseWithCovariance
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy import Context
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    Duration,
    Node,
    PoseWithCovarianceStamped,
    TransformBroadcaster,
    TransformStamped,
    rclpy,
)
from visualization_msgs.msg import Marker

from libracecar.transforms import pose_to_transform, transform_to_pose

from . import deterministic_motion_tracker
from .api import LocalizationBase, localization_params
from .controllers.particles import particles_model, particles_params


@dataclass
class RealNodeConfig:

    n_laser_points: int = 1081

    ground_truth_confidence_threshold: float | None = None

    base_frame: str = "base_link"
    laser_frame: str = "laser_model"
    laser_message_expect_frame: str | None = None

    odom_sub_topic: str = "/vesc/odom"

    visualization_topic: str = "/visualization"

    time_overwrite: bool = False
    invert_odom: bool = False


def _check(a: str, b: str):
    assert a == b, f"not equal: {a} and {b}"


class RealNode(Node):
    def __init__(
        self,
        controller_init: Callable[[localization_params], LocalizationBase],
        *,
        cfg: RealNodeConfig = RealNodeConfig(),
        context: Context | None = None,
    ):
        super().__init__(type(self).__qualname__, context=context)

        self.cfg = cfg

        self.controller_init = controller_init

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.cfg.odom_sub_topic, self.odom_callback, 1
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 1
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1
        )

        self.visualization_pub = self.create_publisher(
            Marker, self.cfg.visualization_topic, 10
        )

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.map = None
        self.tf_odom_laser = None
        self.controller = None

    def do_publish(self):
        if controller := self.get_controller():

            out_laser = controller.get_pose_laser()
            _check(out_laser.child_frame_id, self.cfg.laser_frame)
            out_laser.child_frame_id = "/laser_pf"
            self.tf_broadcaster.sendTransform(out_laser)

            out_odom = controller.get_pose_odom()
            _check(out_odom.child_frame_id, self.cfg.base_frame)
            out_odom.child_frame_id = "/base_link_pf"
            self.tf_broadcaster.sendTransform(out_odom)

            odom_base_link = Odometry(
                header=out_odom.header, child_frame_id=self.cfg.base_frame
            )
            odom_base_link.pose.pose = transform_to_pose(out_odom.transform)
            self.odom_pub.publish(odom_base_link)

    def map_callback(self, map_msg: OccupancyGrid):
        self.map = map_msg
        _ = self.get_controller()

    def get_controller(self) -> LocalizationBase | None:
        if self.controller is not None:
            return self.controller

        if self.tf_odom_laser is None:
            print(
                "self.cfg.base_frame",
                repr(self.cfg.base_frame),
                repr(self.cfg.laser_frame),
            )
            try:
                self.tf_odom_laser = self.tfBuffer.lookup_transform(
                    target_frame=self.cfg.base_frame,
                    source_frame=self.cfg.laser_frame,
                    time=Time(),
                    timeout=Duration(seconds=1),
                )
            except Exception as e:
                print("failed to get transform:", type(e), e)
                return None

        if self.map is None:
            return

        print("get_controller!!!")

        self.controller = self.controller_init(
            localization_params(
                n_laser_points=self.cfg.n_laser_points,
                laser_frame=self.cfg.laser_frame,
                map_frame="map",
                odom_frame=self.cfg.base_frame,
                tf_odom_laser=self.tf_odom_laser,
                map=self.map,
                marker_callback=self.marker_callback,
                ground_truth_callback=self.ground_truth_callback,
            )
        )
        assert isinstance(self.controller, LocalizationBase)
        return self.controller

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # assert msg.header.frame_id == self.cfg.base_frame, (
        #     msg.header.frame_id,
        #     self.cfg.base_frame,
        # )

        if controller := self.get_controller():
            t = TransformStamped()
            t.header = msg.header
            t.child_frame_id = self.cfg.base_frame
            t.transform = pose_to_transform(msg.pose.pose)

            controller.set_pose(t)

    def odom_callback(self, msg: Odometry):
        _check(msg.child_frame_id, self.cfg.base_frame)

        if controller := self.get_controller():
            assert msg.twist.twist.linear.y == 0.0
            if self.cfg.invert_odom:
                msg.twist.twist.linear.x = -msg.twist.twist.linear.x
                msg.twist.twist.angular.z = -msg.twist.twist.angular.z

            # TODO: real messages have no timestamps
            if self.cfg.time_overwrite:
                msg.header.stamp = self.get_clock().now().to_msg()

            controller.odom_callback(msg)

            # print("pose:", controller.get_pose())
            # print("particles:", controller.get_particles())

            self.do_publish()

    def lidar_callback(self, msg: LaserScan):
        _check(
            msg.header.frame_id,
            self.cfg.laser_message_expect_frame or self.cfg.laser_frame,
        )
        if self.cfg.laser_message_expect_frame:
            msg.header.frame_id = self.cfg.laser_frame

        if controller := self.get_controller():

            # TODO: real messages have no timestamps
            msg.header.stamp = self.get_clock().now().to_msg()

            controller.lidar_callback(msg)

            self.do_publish()

    def marker_callback(self, marker: Marker):
        self.visualization_pub.publish(marker)

    def ground_truth_callback(self) -> TransformStamped | None:
        if self.controller is not None:
            pose = self.controller.get_pose_laser()
            conf = self.controller.get_confidence()
            if self.cfg.ground_truth_confidence_threshold is not None:
                if conf > self.cfg.ground_truth_confidence_threshold:
                    return pose
                else:
                    print("not sufficiently confident; not providing a ground truth")


def examplemain():
    """examle (deterministic_motion_tracker)"""
    rclpy.init()
    rclpy.spin(RealNode(deterministic_motion_tracker))
    assert False, "unreachable"


def examplemain2():
    """examle (particles_model)"""
    rclpy.init()
    rclpy.spin(RealNode(particles_model(particles_params(n_particles=1000))))
    assert False, "unreachable"
