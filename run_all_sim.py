#!/usr/bin/env python

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
        deterministic_motion_tracker,
        particles_model,
    )
    from liblocalization.controllers.particles import particles_params
    from libracecar.test_utils import proc_manager

    jax.config.update("jax_platform_name", "cpu")
    np.set_printoptions(precision=5, suppress=True)
    # jax.config.update("jax_enable_x64", True)

    mapdir = Path(__file__).parent / "maps"

    # map = mapdir / "test_map.yaml"
    map = mapdir / "stata_basement.yaml"

    procs = proc_manager.new()

    procs.popen(
        ["rviz2"],
        # env=os.environ | {"LIBGL_ALWAYS_SOFTWARE": "1"},
    )

    procs.ros_node_thread(
        ExampleSimNode,
        particles_model(
            particles_params(use_motion_model=False, plot_level=10, n_particles=500)
        ),
    )
    # procs.ros_node_thread(ExampleSimNode, deterministic_motion_tracker)

    time.sleep(2.0)

    procs.ros_launch("racecar_simulator", "simulate.launch.xml", f"map:={map}")

    procs.popen(
        ["python", "/home/dockeruser/racecar_ws/src/wall_follower_sim/runner.py"]
    )

    procs.spin()


if __name__ == "__main__":
    main()
