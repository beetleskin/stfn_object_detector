#include "GODAlignment.hpp"
#include <stfn_object_detector/Detection2D.h>
#include <stfn_object_detector/Detection2DArr.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/visualization/pcl_visualizer.h>



typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, stfn_object_detector::Detection2DArr> MySyncPolicy;

// globals
shared_ptr<GODAlignment> aligner;
shared_ptr<pcl::visualization::PCLVisualizer> visu;



void init_aligner() {
	//
	// parameters
	// camera matrix
	sensor_msgs::CameraInfo::ConstPtr camera_info_msg_ptr;// = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/rgb/camera_info", ros::Duration(2.0));
	sensor_msgs::CameraInfo camera_info_msg;
	if (!camera_info_msg_ptr) {
		ROS_WARN("BottleSearch: No camera intrinsics available, using default parameters!");
		camera_info_msg.K = { {570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0} };
	} else {
		camera_info_msg = *camera_info_msg_ptr;
	}
	// model file
	std::string model_file = "/home/stfn/testdata/can_1_cloud.pcd";
	ros::param::get("~model_file", model_file);

	//
	// init
	aligner.reset(new GODAlignment(camera_info_msg.K));
	aligner->loadModel(model_file);

	// show model and prepare visu
	visu.reset(new pcl::visualization::PCLVisualizer());
	visu->setBackgroundColor(0.7, 0.7, 0.7);
	visu->addPointCloud<PointT>(aligner->model, pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligner->model, 255, 0, 0), "model");
	visu->addPointCloud<PointT>(aligner->scene, pcl::visualization::PointCloudColorHandlerRGBField<PointT>(aligner->scene), "scene");
	visu->spinOnce();
}


void cluster_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const stfn_object_detector::Detection2DArrConstPtr &det_msg) {
	if (!aligner) {
		ROS_WARN("Aligner not yet initialized, dropping images.");
		return;
	}

	//
	// convert cloud
	PointCloudT::Ptr cloud(new PointCloudT);
	pcl::fromROSMsg (*cloud_msg, *cloud);
	cloud->is_dense = false;
	aligner->setInputCloud(cloud);


	//
	// visu update
	visu->removeAllShapes();
	for (int i = 0; true; ++i) {
		if (!visu->removePointCloud("cluster_" + std::to_string(i)))
			break;
	}
	visu->updatePointCloud<PointT>(cloud, pcl::visualization::PointCloudColorHandlerRGBField<PointT>(cloud), "scene");
	visu->spinOnce();
	if (det_msg->detections.empty()) {
		ROS_INFO("no clusters to check, continuing ...");
		return;
	}
	visu->resetCamera();


	//
	// alignment loop, for each detection
	for (int i = 0; i < det_msg->detections.size(); ++i) {
		const stfn_object_detector::Detection2D &d = det_msg->detections[i];
		ROS_INFO("\tgot detection: [%.1f, %.1f](%.1f, %.1f)", d.leftTop.x, d.leftTop.y, d.bottomRight.x - d.leftTop.x, d.bottomRight.y - d.leftTop.y);

		//
		// extract cluster
		PointCloudT::Ptr cluster_cloud(new PointCloudT());
		//aligner->extract_hypothesis_cluster_crop(aligner->scene, cluster_cloud, d.leftTop.x, d.leftTop.y, d.bottomRight.x - d.leftTop.x, d.bottomRight.y - d.leftTop.y);
		pcl::PointXYZ cluster_center = aligner->extract_hypothesis_cluster_radius(cluster_cloud, d.center.x, d.center.y);
		if (cluster_cloud->empty()) {
			ROS_INFO("\tcluster empty ... aborting");
			continue;
		}

		//
		// debug view
		std::string id = "cluster_" + std::to_string(i);
		visu->addPointCloud<PointT>(cluster_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(cluster_cloud, 0, 255, 0), id);
		visu->addSphere(cluster_center, aligner->model_bounding_sphere_r * 1.5, 0., 1., 1., id);
		visu->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
		// set camera to cluster center
		Eigen::Vector3d world_center_vec(0, 0, 0);
		Eigen::Vector3d cluster_center_vec(cluster_center.x, cluster_center.y, cluster_center.z);
		Eigen::Vector3d cluster_world_vec = (world_center_vec - cluster_center_vec).normalized();
		Eigen::Vector3d cam_pos = cluster_center_vec + 0.3 * cluster_world_vec;
		visu->setCameraPosition(cam_pos[0], cam_pos[1], cam_pos[2], cluster_center_vec[0], cluster_center_vec[1], cluster_center_vec[2], 0, -1, 0);
		visu->spinOnce();
		ROS_INFO("\textraced cloud cluster size: %i", (int)cluster_cloud->size());


		//
		// model-cluster alignment
		PointCloudT::Ptr model_aligned(new PointCloudT);
		Eigen::Matrix4f object_cam_transform_matrix;
		bool success = aligner->align_cloud_to_model(cluster_cloud, object_cam_transform_matrix, model_aligned);

		if(success) {
			Eigen::Affine3f object_cam_transform(object_cam_transform_matrix);
			Eigen::Quaternion<float> r(object_cam_transform.rotation());
			Eigen::Vector3f t(object_cam_transform.translation());
		}
		visu->updatePointCloud<PointT>(model_aligned,  pcl::visualization::PointCloudColorHandlerCustom<PointT>(model_aligned, 255, 0, 0), "model");


		ROS_WARN("Using only first detection!");
		break;
	}

	visu->spin();
}





int main(int argc, char **argv) {
	ros::init(argc, argv, "aligner");

	ros::NodeHandle nh;

	init_aligner();


	// register synced subscribers
	message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "tmp_cloud", 1);
	message_filters::Subscriber<stfn_object_detector::Detection2DArr> det_sub(nh, "dhf_detections", 1);
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pcl_sub, det_sub);
	sync.registerCallback(bind(&cluster_callback, _1, _2 ));


	ros::spin();
	return 0;
}