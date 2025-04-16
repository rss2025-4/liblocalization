#!/usr/bin/env python

import os

from libracecar.sandbox import isolate


@isolate
def main():
    import time
    from pathlib import Path

    import better_exceptions
    import jax
    import numpy as np

    from liblocalization import (
        ExampleSimNode,
        RealNode,
        deterministic_motion_tracker,
        models_base_dir,
        particles_model,
        particles_params,
        stats_base_dir,
    )
    from liblocalization.controllers.stats import stats_params
    from liblocalization.real_node import RealNodeConfig
    from libracecar.test_utils import proc_manager

    jax.config.update("jax_platform_name", "cpu")
    np.set_printoptions(precision=5, suppress=True)
    # jax.config.update("jax_enable_x64", True)

    mapdir = Path(__file__).parent / "maps"

    # map = mapdir / "test_map.yaml"
    map = mapdir / "stata_basement.yaml"

    bag_dir = Path("/home/alan/6.4200/lab 6 rosbags/")
    # bag = "rrt_angle_longer"
    bag = "rrt_angle"

    # for bag in [
    #     "astar_angle_2",
    #     # "astar_angle_longer",
    #     "astar_longer",
    #     "rrt_angle",
    #     "rrt_angle_longer",
    #     "rrtstar_longer",
    #     "rrtstar_longer_sb_larger",
    # ]:

    # bag_dir = Path(
    #     "/home/alan/6.4200/Localization Bags - 04012025/localization_testers"
    # )
    # bag = "red_cciclee_down"

    procs = proc_manager.new()
    procs.spin_thread()

    procs.popen(
        ["rviz2"],
        env=os.environ | {"LIBGL_ALWAYS_SOFTWARE": "1"},
    )

    # procs.popen(
    #     ["emacs"],
    #     cwd="/home/alan/6.4200/Localization Bags - 04012025/localization_testers",
    # )

    procs.ros_node_thread(
        lambda context: RealNode(
            particles_params(
                plot_level=10,
                n_particles=500,
                # use_motion_model=False,
                # stats_in_dir=stats_base_dir / "sim",
                stats_in_dir=None,
                stats_out_dir=stats_base_dir / "rosbags_lidar_fixed2",
                # evidence_factor=1.0,
                model_path=models_base_dir / "model2.pkl",
            ),
            cfg=RealNodeConfig(
                laser_frame="laser_model",
            ),
            context=context,
        ),
    )

    time.sleep(3.0)

    procs.ros_launch(
        "racecar_simulator", "localization_simulate.launch.xml", f"map:={map}"
    )

    ### for maps
    # procs.ros_run(
    #     "nav2_map_server", "map_server", ros_params={"yaml_filename": str(map)}
    # )

    # procs.ros_launch("racecar_simulator", "racecar_model.launch.xml")

    # procs.ros_run("racecar_simulator", "simulate")

    # procs.ros_run(
    #     "nav2_lifecycle_manager",
    #     "lifecycle_manager",
    #     ros_params={"autostart": "True", "node_names": "['map_server']"},
    # )

    ### simulation + wall follower
    # procs.ros_launch("racecar_simulator", "simulate.launch.xml", f"map:={map}")
    # procs.popen(
    #     ["python", "/home/dockeruser/racecar_ws/src/wall_follower_sim/runner.py"]
    # )

    # wait for compile
    time.sleep(10)

    bag_p = procs.popen(
        ["ros2", "bag", "play", bag],
        cwd=str(bag_dir),
    )

    time.sleep(10000)


if __name__ == "__main__":
    main()
