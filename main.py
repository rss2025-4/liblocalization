#!/usr/bin/env python


import jax
import numpy as np
import rclpy

from liblocalization import (
    RealNode,
    models_base_dir,
    particles_params,
    stats_base_dir,
)
from liblocalization.real_node import RealNodeConfig

jax.config.update("jax_platform_name", "cpu")
np.set_printoptions(precision=5, suppress=True)

# jax.config.update("jax_enable_x64", True)


def main():
    rclpy.init(args=[])
    node = RealNode(
        particles_params(
            plot_level=10,
            n_particles=500,
            # use_motion_model=False,
            # stats_in_dir=stats_base_dir / "sim",
            stats_in_dir=None,
            stats_out_dir=stats_base_dir / "real",
            # evidence_factor=1.0,
            model_path=models_base_dir / "model2.pkl",
        ),
        cfg=RealNodeConfig(
            laser_frame="laser_model",
        ),
    )
    rclpy.spin(node)


if __name__ == "__main__":
    main()
