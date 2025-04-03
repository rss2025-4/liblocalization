from dataclasses import dataclass
from typing import Protocol

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformStamped


class Controller(Protocol):
    def init(self, map: OccupancyGrid, init_pos: TransformStamped): ...
    def on_laser(self, msg: LaserScan, /) -> None: ...
    def on_odom(self, msg: Odometry, /) -> None: ...
    def get_location_est(self) -> TransformStamped: ...


class DummyController:
    def init(self, map: OccupancyGrid, init_pos: TransformStamped):
        return None

    def on_laser(self, msg: LaserScan, /):
        return None

    def on_odom(self, msg: Odometry, /):
        return None

    def get_location_est(self) -> TransformStamped:
        return TransformStamped()


@dataclass
class EvalRes:
    # TODO
    pass


def do_eval(c: Controller, *blah) -> EvalRes:
    # eval people do this
    raise NotImplementedError()


# eval people do this
class Node:
    pass


def test_eval():
    do_eval(DummyController())
