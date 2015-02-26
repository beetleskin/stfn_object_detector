#include "GODDetection.hpp"
#include "GODMapping.hpp"
#include "myutils.hpp"

#include <stfn_object_detector/Detection2D.h>
#include <stfn_object_detector/Detection2DArr.h>

#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <visualization_msgs/ImageMarker.h>

#include <vector>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;


using namespace std;


GODDetection::Ptr detector;
GODMapping::Ptr mapper;
const float COLORS[9] = { 0,0,1, 0,1,0, 1,0,0};
ros::Publisher detections_pub, tmp_cloud_pub;



void pub_detections(const vector<Candidate> &candidates, const std_msgs::Header &input_img_header) {
	stfn_object_detector::Detection2DArr detectionArr;
	
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {
		// create detection
		stfn_object_detector::Detection2D d;
		d.id = candidates[candNr].class_id;
		d.confidence = candidates[candNr].confidence;
		d.center.x = candidates[candNr].center.x;
		d.center.y = candidates[candNr].center.y;
		d.leftTop.x = candidates[candNr].bb[0].x;
		d.leftTop.y = candidates[candNr].bb[0].y;
		d.bottomRight.x = candidates[candNr].bb[1].x;
		d.bottomRight.y = candidates[candNr].bb[1].y;
		detectionArr.detections.push_back(d);
	}

	// publish detections
	detectionArr.header.stamp = input_img_header.stamp;
	detectionArr.header.frame_id = input_img_header.frame_id;
	detections_pub.publish(detectionArr);
}





void imageCallback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg) {
	ROS_INFO("receiving data package ...");
	
	if (!detector) {
		ROS_WARN("Detector not yet initialized, dropping images.");
		return;
	}

	cv_bridge::CvImagePtr rgb_cv_ptr, depth_cv_ptr;
	try {
		rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
		depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
	} catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	
	// TODO: depending on the source, the depth image comes in mm or m. training is done on mm.
	if(true)
		depth_cv_ptr->image *= 1000;


	//
	// run the full detection
	vector<Candidate> candidates;
	detector->detect(rgb_cv_ptr->image, depth_cv_ptr->image, candidates);

	// publish visualization markers
	ROS_INFO("got %d detections", (int)candidates.size());
	pub_detections(candidates, rgb_msg->header);


	//
	// TODO: publish pointcloud (this is a temporary workaround for the aligner)
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	calcPointsRGBPCL(depth_cv_ptr->image, rgb_cv_ptr->image, cloud, 1.0);
	cloud->header.stamp = rgb_msg->header.stamp.toNSec()/1e3;
	cloud->header.frame_id = rgb_msg->header.frame_id;
	tmp_cloud_pub.publish(cloud);


	//
	// run the mapping and persistence
	//mapper->update(depth_cv_ptr->image, cloud, candidates);
}




int main(int argc, char **argv) {
	ros::init(argc, argv, "detector");

	// initialize detector
	detector.reset(new GODDetection);
	mapper.reset(new GODMapping);

	// register publishers
	ros::NodeHandle nh;
	detections_pub = nh.advertise<stfn_object_detector::Detection2DArr>("dhf_detections", 10);
	tmp_cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("tmp_cloud", 10);

	// register synced subscribers
	image_transport::ImageTransport it(nh);
	image_transport::SubscriberFilter rgb_sub( it, "/camera/rgb/image", 1);
	image_transport::SubscriberFilter depth_sub( it, "/camera/depth/image", 1);
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
	sync.registerCallback(bind(&imageCallback, _1, _2 ));


	ros::spin();
	return 0;
}