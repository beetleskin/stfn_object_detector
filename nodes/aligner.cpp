#include <stfn_object_detector/Detection2D.h>
#include <stfn_object_detector/Detection2DArr.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/point_tests.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>


using namespace std;


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNormalT;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, stfn_object_detector::Detection2DArr> MySyncPolicy;

// globals
PointCloudT::Ptr model_cloud;
pcl::PointCloud<PointNormalT>::Ptr model_cloud_normals;
shared_ptr<pcl::visualization::PCLVisualizer> p;




void clip_pointcloud(PointCloudT::Ptr &cloud, PointCloudT::Ptr &coud_clip, int off_x, int off_y, int width, int height) {
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	off_x -= width / 2;
	off_y -= height / 2;
	width *= 2;
	height *= 2;
	if (off_x < 0)
		off_x = 0;
	if (off_y < 0)
		off_y = 0;
	if (off_x + width >= cloud->width)
		width = cloud->width - off_x - 1;
	if (off_y + height >= cloud->height)
		height = cloud->height - off_y - 1;

	for (int y = off_y; y < off_y + height; ++y) {
		for (int x = off_x; x < off_x + width; ++x) {
			int i = y * cloud->width + x;
			if (pcl::isFinite (cloud->points[i]))
				inliers->indices.push_back(i);
		}
	}

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (inliers);
	extract.setNegative (false);
	extract.filter (*coud_clip);
}


void det_pcl_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const stfn_object_detector::Detection2DArrConstPtr &det_msg) {
	pcl::PCLPointCloud2 pcl_pc;
	PointCloudT::Ptr cloud(new PointCloudT);
	pcl_conversions::toPCL(*cloud_msg, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, *cloud);

	p->removeAllPointClouds();
	//pcl::visualization::PointCloudColorHandlerRGBField<PointT> ch_rgb_cloud(cloud);
	//p->addPointCloud(cloud, ch_rgb_cloud, "cloud");
	//p->spinOnce();




	for (int i = 0; i < det_msg->detections.size(); ++i) {
		const stfn_object_detector::Detection2D &d = det_msg->detections[i];
		ROS_INFO("got detection: [%.1f, %.1f](%.1f, %.1f)", d.leftTop.x, d.leftTop.y, d.bottomRight.x - d.leftTop.x, d.bottomRight.y - d.leftTop.y);

		PointCloudT::Ptr detection_cloud(new PointCloudT);
		PointCloudT::Ptr model_registered(new PointCloudT);
		pcl::PointCloud<PointNormalT>::Ptr detection_cloud_normals(new pcl::PointCloud<PointNormalT>);
		clip_pointcloud(cloud, detection_cloud, d.leftTop.x, d.leftTop.y, d.bottomRight.x - d.leftTop.x, d.bottomRight.y - d.leftTop.y);
		ROS_INFO("clipped cloud: (%d points)", detection_cloud->points.size());
		pcl::io::savePCDFile ("/home/stfn/clip.pcd", *detection_cloud);
		return;
		// subsample
		pcl::VoxelGrid<PointT> gridFilter;
		gridFilter.setLeafSize (0.005, 0.005, 0.005);
		gridFilter.setInputCloud (detection_cloud);
		gridFilter.filter (*detection_cloud);
		ROS_INFO("voxel_grid: (%d points)", detection_cloud->points.size());
		

		// normals
		pcl::NormalEstimationOMP<PointT, PointNormalT> norm_est;
		pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
		norm_est.setSearchMethod (tree);
		norm_est.setRadiusSearch (0.015);
		norm_est.setInputCloud (detection_cloud);
		norm_est.compute (*detection_cloud_normals);
		ROS_INFO("normals");

		// align
		pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;
		//icp.setRANSACOutlierRejectionThreshold(0.005);
		//icp.setMaxCorrespondenceDistance(0.01);
		icp.setTransformationEpsilon (1e-6);
		icp.setMaxCorrespondenceDistance (0.01);
		icp.setInputSource (model_cloud);
		icp.setInputTarget (detection_cloud);
		icp.align(*model_registered);

		Eigen::Matrix4f transformation = icp.getFinalTransformation ();

		p->removePointCloud("detection_bb");
		p->removePointCloud("model_registered");
		pcl::visualization::PointCloudColorHandlerRGBField<PointT> ch_detection_cloud (detection_cloud);
		pcl::visualization::PointCloudColorHandlerCustom<PointT> ch_model_cloud (detection_cloud, 255, 255, 0);
		p->addPointCloud (detection_cloud, ch_detection_cloud, "detection_bb");
		p->addPointCloud (model_registered, ch_model_cloud, "model_registered");
		p->spin();


		/*
		Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
		PointCloudWithNormals::Ptr icp_result = points_with_normals_src;
		reg.setMaximumIterations (2);
		for (int i = 0; i < 30; ++i) {
		    PCL_INFO ("Iteration Nr. %d.\n", i);

		    // save cloud for visualization purpose
		    points_with_normals_src = reg_result;

		    // Estimate
		    reg.setInputSource (points_with_normals_src);
		    reg.align (*reg_result);

		    //accumulate transformation between each Iteration
		    Ti = reg.getFinalTransformation () * Ti;

		    //if the difference between this transformation and the previous one
		    //is smaller than the threshold, refine the process by reducing
		    //the maximal correspondence distance
		    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
		        reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);

		    prev = reg.getLastIncrementalTransformation ();

		    // visualize current state
		    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
		}

		//
		// Get the transformation from target to source
		targetToSource = Ti.inverse();*/
	}


}


void initAligner() {
	string model_file = "/home/stfn/testdata/can_1_cloud.pcd";
	ros::param::get("~model_file", model_file);


	// init
	model_cloud.reset(new PointCloudT);
	model_cloud_normals.reset(new pcl::PointCloud<PointNormalT>);
	if (pcl::io::loadPCDFile<PointT>(model_file, *model_cloud) == -1) {
		ROS_ERROR ("Couldn't read model file: %s", model_file.c_str());
		return;
	}
	ROS_INFO("model loaded: (%d points)", model_cloud->points.size());


	// subsample
	pcl::VoxelGrid<PointT> gridFilter;
	gridFilter.setLeafSize (0.005, 0.005, 0.005);
	gridFilter.setInputCloud (model_cloud);
	gridFilter.filter(*model_cloud);
	ROS_INFO("model voxel grid: (%d points)", model_cloud->points.size());

	// normals
	pcl::NormalEstimationOMP<PointT, PointNormalT> norm_est;
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	norm_est.setSearchMethod (tree);
	norm_est.setRadiusSearch (0.015);
	norm_est.setInputCloud (model_cloud);
	norm_est.compute (*model_cloud_normals);

	p.reset(new pcl::visualization::PCLVisualizer ("ICP"));
	p->setBackgroundColor(0.7, 0.7, 0.7);
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> ch_rgb_model(model_cloud);
	p->addPointCloud (model_cloud, ch_rgb_model, "model");
	p->spin();
}


int main(int argc, char **argv) {
	ros::init(argc, argv, "aligner");

	ros::NodeHandle nh;

	initAligner();

	// register synced subscribers
	message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "tmp_cloud", 1);
	message_filters::Subscriber<stfn_object_detector::Detection2DArr> det_sub(nh, "dhf_detections", 1);
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pcl_sub, det_sub);
	sync.registerCallback(bind(&det_pcl_callback, _1, _2 ));


	ros::spin();
	return 0;
}