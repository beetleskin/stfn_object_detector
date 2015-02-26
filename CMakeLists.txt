cmake_minimum_required(VERSION 2.8.3)
project(stfn_object_detector)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

#########################
## TODO TODO TODO TODO ##
#########################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++0x")

# custom OpenCV
set(OpenCV_INCLUDE_DIRS "/home/stfn/libs/opencv-2.4.10/include" "/home/stfn/libs/opencv-2.4.10/include/opencv" "/home/stfn/libs/opencv-2.4.10/include/opencv2" "/home/stfn/libs/opencv-2.4.10/modules/*/include")
set(OpenCV_LIBRARY_DIRS "/home/stfn/libs/opencv-2.4.10/build/lib")
set(OpenCV_LIBS "opencv_core;opencv_highgui;opencv_imgproc;opencv_gpu;opencv_objdetect")
link_directories(${OpenCV_LIBRARY_DIRS})

# custom Eigen3
set(EIGEN_INCLUDE_DIRS "/home/stfn/libs/eigen3.2.1")

# hf_mc
set(DHF_DIR "/home/stfn/dev/hf_mc/evaluations/rewrite/detector")
#file(GLOB DHF_SOURCES ${DHF_DIR}/*.cpp)
set(DHF_SOURCES 
	${DHF_DIR}/CRForest.cpp
	${DHF_DIR}/CRForestDetector.cpp
	${DHF_DIR}/CRPatch.cpp
	${DHF_DIR}/CRTree.cpp
	${DHF_DIR}/HoG.cpp
	src/GODImageMemory.cpp
)

#########################




## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
	pcl_conversions
	pcl_ros
	roscpp
	rospy
	sensor_msgs
	std_msgs
	cv_bridge
	image_transport
	message_filters
	message_generation
	visualization_msgs
	message_generation
	tf_conversions
	eigen_conversions
)

# custom PCL
#find_package(PCL 1.8.0 REQUIRED)
#link_directories(${PCL_LIBRARY_DIRS})


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
	FILES
	Detection2D.msg
	Detection2DArr.msg
	Detection3D.msg
	Detection3DArr.msg
)

## Generate services in the 'srv' folder
add_service_files(
	FILES
	DetectWithPose.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
	DEPENDENCIES
	std_msgs
	geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES stfn_object_detector
#  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp rospy sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
#	${PCL_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
	${DHF_DIR}
	${BOOST_INCLUDE_DIRS_156}
)

## Declare a cpp library
# add_library(stfn_object_detector
#   src/${PROJECT_NAME}/stfn_object_detector.cpp
# )

## Declare a cpp executable
#add_executable(detector ${DHF_DIR}/CRForest-Detector.cpp ${DHF_SOURCES})
add_executable(ros_detector nodes/detector.cpp src/GODDetection.cpp src/GODMapping.cpp ${DHF_SOURCES})
add_executable(ros_trainer nodes/trainer.cpp src/GODTraining.cpp ${DHF_SOURCES})
add_executable(ros_aligner2 nodes/aligner2.cpp src/GODAlignment.cpp)
add_executable(ros_mapper nodes/mapper.cpp)
add_executable(ros_detect_with_pose_service src/GODAlignment.cpp src/GODDetection.cpp src/detect_with_pose_service.cpp ${DHF_SOURCES})
add_executable(ros_aligner_fitness_checker test/aligner_fitness_checker.cpp src/GODAlignment.cpp)

add_dependencies(ros_detector stfn_object_detector_generate_messages_cpp)
add_dependencies(ros_aligner2 stfn_object_detector_generate_messages_cpp)
add_dependencies(ros_mapper stfn_object_detector_generate_messages_cpp)
add_dependencies(ros_detect_with_pose_service stfn_object_detector_generate_messages_cpp)



## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(stfn_object_detector_node stfn_object_detector_generate_messages_cpp)

## Specify libraries to link a library or executable target against
#target_link_libraries(detector
#	${catkin_LIBRARIES}
#	${OpenCV_LIBS}
#	tbb
#)

target_link_libraries(ros_detector
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	tbb
	libvtkCommon.so
	libvtkFiltering.so
	libvtkGraphics.so
)

target_link_libraries(ros_trainer
	${catkin_LIBRARIES}
	tbb
)

target_link_libraries(ros_mapper
	${catkin_LIBRARIES}
)

target_link_libraries(ros_aligner2
	${catkin_LIBRARIES}
	libvtkCommon.so
	libvtkFiltering.so
	libvtkGraphics.so
)

target_link_libraries(ros_detect_with_pose_service
	${catkin_LIBRARIES}
	tbb
)

target_link_libraries(ros_aligner_fitness_checker
#	${PCL_LIBS}
	${catkin_LIBRARIES}
	libvtkCommon.so
	libvtkFiltering.so
	libvtkGraphics.so
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS stfn_object_detector stfn_object_detector_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_stfn_object_detector.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
