#!/bin/bash -l

source /opt/ros/$ROS_DISTRO/setup.bash

set -eux

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
cd $SCRIPT_DIR

direnv allow
eval "$(direnv export bash)"

make clean
make html
