#include "Candidate.hpp"
#include "GODAlignment.hpp"
#include "GODDetection.hpp"
#include "myutils.hpp"

#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/image_encodings.h>

#include "eigen_conversions/eigen_msg.h"


#include <stfn_object_detector/DetectWithPose.h>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;




class DetectWithPoseService {

public:
	DetectWithPoseService(ros::NodeHandle& nh) : nh(nh) {


		//
		// create synchronized connections to image topics

		it.reset(new image_transport::ImageTransport(nh));
		rgb_sub.reset(new image_transport::SubscriberFilter);
		depth_sub.reset(new image_transport::SubscriberFilter);
		sync.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *rgb_sub, *depth_sub));
		sync->registerCallback(bind(&DetectWithPoseService::imageCallback, this, _1, _2 ));
		model_aligned_debug_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("tmp_cloud", 10);


		//
		// create detector

		detector.reset(new GODDetection);


		//
		// create aligner

		sensor_msgs::CameraInfo::ConstPtr camera_info_msg_ptr = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/rgb/camera_info", ros::Duration(2.0));
		sensor_msgs::CameraInfo camera_info_msg;
		if (!camera_info_msg_ptr) {
			ROS_WARN("GODAlignment: No camera intrinsics available, using default parameters!");
			camera_info_msg.K = { {570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0} };
		} else {
			camera_info_msg = *camera_info_msg_ptr;
		}

		// model file
		// TODO: For more than one model to align, this needs to be adapted. Right now only one model is loaded.
		std::string model_file = "/home/recognition_database/models/can_model.pcd";
		ros::param::get("~model_file", model_file);

		// init
		aligner.reset(new GODAlignment(camera_info_msg.K));
		aligner->loadModel(model_file);
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
		// run the full detection (2D)

		ROS_INFO("starting HRF detection ...");
		vector<Candidate> candidates;
		detector->detect(rgb_image->image, depth_image->image, candidates);
		ROS_INFO("found %d detection candidates", (int)candidates.size());


		if(candidates.empty()) {
			return true;
		}


		//
		// run the alignment (3D)

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		calcPointsRGBPCL(depth_image->image, rgb_image->image, cloud, 1.0);
		aligner->setInputCloud(cloud);

		// align each 2D detection
		for (int i = 0; i < candidates.size(); ++i) {
			const Candidate &cand = candidates[i];


			// only examine candidates that belong to the requested model id resp. class name
			if(cand.class_name != req.model_id)
				continue;

			// extract a subcloud around the candidates center
			PointCloudT::Ptr cluster_cloud(new PointCloudT());
			pcl::PointXYZ cluster_center = aligner->extract_hypothesis_cluster_radius(cluster_cloud, cand.center.x, cand.center.y);
			
			if (cluster_cloud->empty()) {
				ROS_INFO("\tcluster empty ... aborting");
				continue;
			}



			//
			// model-cluster alignment

			PointCloudT::Ptr model_aligned(new PointCloudT);
			Eigen::Matrix4f object_cam_transform_matrix;
			bool success = aligner->align_cloud_to_model(cluster_cloud, object_cam_transform_matrix, model_aligned);

			if(!success) {
				ROS_WARN("Could not align detection, continuing with next ...");
				continue;
			}

			// build affine transofmation
			Eigen::Affine3d object_cam_transform(object_cam_transform_matrix.cast<double>());
			//Eigen::Quaternion<double> r(object_cam_transform.rotation());
			//Eigen::Vector3d t(object_cam_transform.translation());

			// publish aligned model as pointcloud
			model_aligned->header.stamp = rgb_image->header.stamp.toNSec()/1e3;
			model_aligned->header.frame_id = rgb_image->header.frame_id;
			model_aligned_debug_pub.publish(*model_aligned);

			// generate detection response
			stfn_object_detector::Detection3D detection;
			detection.model_id = cand.class_id;
			detection.confidence = cand.confidence;
			tf::poseEigenToMsg(object_cam_transform, detection.pose);
			res.detections.detections.push_back(detection);
			

			// TODO: for now only the candidate with the highest confidence is examined for alignment
			break;
		}


		return true;
	};


	/**
	*	Grabs one image from both, rgb and depth subscribers, then deregisteres itselfes as we only need one
	* 	image for the detection.
	*	TODO: Do I need a lock/semaphore in case the imagecallback is called more often than once?
	*/
	void imageCallback(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg) {
		try {
			rgb_image = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
			depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
			depth_image->image *= 1000;
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
	GODAlignment::Ptr aligner;
	cv_bridge::CvImagePtr rgb_image;
	cv_bridge::CvImagePtr depth_image;

	ros::NodeHandle& nh;
	shared_ptr<image_transport::ImageTransport> it;
	shared_ptr<image_transport::SubscriberFilter> rgb_sub;
	shared_ptr<image_transport::SubscriberFilter> depth_sub;
	shared_ptr<message_filters::Synchronizer<MySyncPolicy> > sync;
	ros::Publisher model_aligned_debug_pub;
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
