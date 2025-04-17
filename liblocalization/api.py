import abc
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    TransformStamped,
)
from visualization_msgs.msg import Marker

from libracecar.utils import check_eq


@dataclass(frozen=True)
class localization_params:
    map_frame: str
    odom_frame: str
    laser_frame: str

    tf_odom_laser: TransformStamped

    #: the map image and origin
    map: OccupancyGrid

    #: length of "ranges" in LaserScan; needed for jax to compile computation in advance.
    n_laser_points: int = 100

    #: callback that can be called from the controller to accept a Marker.
    #:
    #: By default, does nothing with the Marker.
    marker_callback: Callable[[Marker], None] = lambda _: None

    #: callback that can be called from the controller to know the sim ground truth.
    #:
    #: for debugging purposes.
    #:
    #: By default, returns None
    ground_truth_callback: Callable[[], TransformStamped | None] = lambda: None


class LocalizationBase(abc.ABC):
    """common interface implemented by localization algorithms"""

    #: shared parameters
    cfg: localization_params

    @abstractmethod
    def odom_callback(self, msg: Odometry) -> None:
        """
        update the algorithm with an odometry on self.cfg.laser_frame.

        only twist will be used.

        must have a correct timestamp.
        """

        # not needed: we are only using twist which is in "child_frame_id"
        # assert msg.header.frame_id == self.cfg.map_frame

        check_eq(msg.child_frame_id, self.cfg.odom_frame)

    @abstractmethod
    def lidar_callback(self, msg: LaserScan) -> None:
        """
        update the algorithm with  a scan on self.cfg.laser_frame.

        must have a correct timestamp.
        """
        check_eq(msg.header.frame_id, self.cfg.laser_frame)
        assert (
            len(msg.ranges) == self.cfg.n_laser_points
        ), f"expected {self.cfg.n_laser_points} laser points, got {len(msg.ranges)}"

    @abstractmethod
    def set_pose(self, pose: TransformStamped) -> None:
        """
        set the initial pose.

        pose is given as a transform of self.cfg.map_frame -> self.cfg.odom_frame
        """
        check_eq(pose.header.frame_id, self.cfg.map_frame)
        check_eq(pose.child_frame_id, self.cfg.odom_frame)

    @abstractmethod
    def get_pose_laser(self) -> TransformStamped:
        """
        get the current best pose estimate, in self.cfg.laser_frame
        """
        ...

    @abstractmethod
    def get_pose_odom(self) -> TransformStamped:
        """
        get the current best pose estimate, in self.cfg.odom_frame
        """
        ...

    @abstractmethod
    def get_particles(self) -> np.ndarray:
        """
        get the particles as a (n, 3) array with (x, y, theta)
        in self.cfg.laser_frame
        """
        ...

    def get_confidence(self) -> float:
        return -100000.0
