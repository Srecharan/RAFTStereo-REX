cmake_minimum_required(VERSION 3.0.2)
project(raft_stereo_leaf)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  yoloV8_seg
)

# Install Python modules
catkin_python_setup()

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  depth.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    message_filters
    roscpp
    rospy
    sensor_msgs
    std_msgs
    yoloV8_seg
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Mark executable scripts for installation
install(PROGRAMS
  ros/nodes/raft_stereo_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)