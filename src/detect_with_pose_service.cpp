#include "GODDetection.hpp"
#include "myutils.hpp"

#include "ros/ros.h"

#include <stfn_object_detector/DetectWithPose.h>


bool detect_with_pose(stfn_object_detector::DetectWithPose::Request &req, stfn_object_detector::DetectWithPose::Response &res) {
	
	return true;
}


int main(int argc, char **argv){
	ros::init(argc, argv, "god_detect_with_pose_server");
	ros::NodeHandle n;

	ros::ServiceServer service = n.advertiseService("detect_with_pose", detect_with_pose);
	ROS_INFO("GOD detection service ready.");
	ros::spin();

	return 0;
}
