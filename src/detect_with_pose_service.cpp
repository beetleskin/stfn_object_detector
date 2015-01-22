#include "GODDetection.hpp"
#include "myutils.hpp"

#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>


#include <stfn_object_detector/DetectWithPose.h>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;




class DetectWithPoseService {

public:
	DetectWithPoseService(ros::NodeHandle& nh) : nh(nh) {

		// create synchronized connections to image topics
		it.reset(new image_transport::ImageTransport(nh));
		rgb_sub.reset(new image_transport::SubscriberFilter);
		depth_sub.reset(new image_transport::SubscriberFilter);
		sync.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *rgb_sub, *depth_sub));
		sync->registerCallback(bind(&DetectWithPoseService::imageCallback, this, _1, _2 ));

		// create detector
		detector.reset(new GODDetection);
	};

	~DetectWithPoseService() {

	};


	bool detect_with_pose_callback(stfn_object_detector::DetectWithPose::Request &req, stfn_object_detector::DetectWithPose::Response &res) {
		
		ROS_INFO("DetectWithPoseService called");

		// 
		// wait for images
		ROS_INFO("waiting for images ...");
		enableImageCallback();
		ros::Rate r(10);
		while(!rgb_image || !depth_image) {
			ros::spinOnce();
			r.sleep();
		}


		//
		// run the full detection
		ROS_INFO("starting HRF detection ...");
		vector<vector<float> > candidates;
		vector<vector<cv::Point2f> > boundingboxes;
		detector->detect(rgb_image->image, depth_image->image, candidates, boundingboxes);

		ROS_INFO("found %d detection candidates", (int)candidates.size());

		return true;
	};


	/**
	*	Grabs one image from both, rgb and depth subscribers, then deregisteres itselfes as we only need one
	* 	image for the detection.
	*	TODO: Do I need a lock/semaphore in case the imagecallback is called more often than once?
	*/
	void imageCallback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg) {
		ROS_INFO("got rgb and depth image");
		try {
			rgb_image = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
			depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
		} catch (cv_bridge::Exception &e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
		}


		disableImageCallback();
	}


private:
	void enableImageCallback() {
		rgb_sub->subscribe(*it, "/camera/rgb/image", 1);
		depth_sub->subscribe(*it, "/camera/depth/image", 1);
	}


	void disableImageCallback() {
		rgb_sub->unsubscribe();
		depth_sub->unsubscribe();
	}


private:
	GODDetection::Ptr detector;
	cv_bridge::CvImagePtr rgb_image;
	cv_bridge::CvImagePtr depth_image;

	ros::NodeHandle& nh;
	shared_ptr<image_transport::ImageTransport> it;
	shared_ptr<image_transport::SubscriberFilter> rgb_sub;
	shared_ptr<image_transport::SubscriberFilter> depth_sub;
	shared_ptr<message_filters::Synchronizer<MySyncPolicy> > sync;
};









int main(int argc, char **argv){
	ros::init(argc, argv, "god_detect_with_pose_server");


	// create and advertise service
	ros::NodeHandle nh;
	DetectWithPoseService dwps(nh);
	ros::ServiceServer service = nh.advertiseService("detect_with_pose", &DetectWithPoseService::detect_with_pose_callback, &dwps);

	// ready to detect!
	ROS_INFO("DetectWithPoseService ready.");
	ros::spin();

	return 0;
}