#!/usr/bin/env python

from libracecar.sandbox import isolate


@isolate
def main():
    import time
    from pathlib import Path

    from liblocalization.main import Localization, localization_config
    from libracecar.test_utils import proc_manager

    mapdir = Path(__file__).parent / "maps"

    # map = mapdir / "test_map.yaml"
    map = "/home/alan/6.4200/racecar_simulator/maps/stata_basement.yaml"

    procs = proc_manager.new()

    procs.popen(
        ["rviz2"],
        # env=os.environ | {"LIBGL_ALWAYS_SOFTWARE": "1"},
    )

    # procs.ros_node_subproc(Localization, localization_config())
    procs.ros_node_thread(Localization, localization_config())

    time.sleep(1.0)

    procs.ros_launch("racecar_simulator", "simulate.launch.xml", f"map:={map}")

    procs.popen(
        ["python", "/home/dockeruser/racecar_ws/src/wall_follower_sim/runner.py"]
    )

    procs.spin()


if __name__ == "__main__":
    main()
