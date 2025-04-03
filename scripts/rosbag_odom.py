from pathlib import Path

from rosbags.highlevel import AnyReader

# CONE_LOCATION_MSG = """
# float32 x_pos
# float32 y_pos
# """
# # from rosbags.typesys.types import vs_msgs__msg__ConeLocation as ConeLocation
# ## TODO add bagpath here
# # bagpath = Path("/Users/aa/rosbags/rosbags/rosbags_lab3/rosbag_point5_speed_point5_dist")
# # bagpath = Path(
# #     "/Users/aa/rosbags/rosbags/rosbags_lab3/rosbag_onepointo_speed_point5_dist_left_withscore_2"
# # )
# # bagpath = Path("/Users/aa/rosbags/lab4_tests/lab4_line_5")
# # file = "lab4_line_3"  # "line_0316" #
# # bagpath = Path(f"/Users/aa/rosbags/lab4_tests/{file}")
bagpath = Path(
    "/home/alan/6.4200/Localization Bags - 04012025/localization_testers/bottom_bathroom_right_door/"
)


# typestore = get_typestore(Stores.ROS2_HUMBLE)  # ros humble


# # register_types(get_types_from_msg(
# #         ConeLocation, 'vs_msgs/msg/ConeLocation'))
# typestore.register(get_types_from_msg(CONE_LOCATION_MSG, "vs_msgs/msg/ConeLocation"))
# ConeLocation = typestore.types["vs_msgs/msg/ConeLocation"]


# print(os.path.exists(bagpath))  # make sure path exists

# setpoint = 0.5  # distance setpoint


def get_data_from_bag():
    topic = "/odom"
    # use rosbags AnyReader to read bag at a certain path
    with AnyReader([bagpath]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                print(msg.twist.covariance)
                print()
                print()
            except:
                pass
