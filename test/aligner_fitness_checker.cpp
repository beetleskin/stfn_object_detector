#include "GODAlignment.hpp"
#include "myutils.hpp"

#include <boost/foreach.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>



using namespace std;
using namespace boost;


int main(int argc, char **argv) {
	ros::init(argc, argv, "aligner_optimization", ros::init_options::AnonymousName);

	// disable cout and cerr and everything
	ostringstream oss;
	streambuf* oldCoutStreamBuf = cout.rdbuf();
	cout.rdbuf( oss.rdbuf() );
	streambuf* oldCerrStreamBuf = cerr.rdbuf();
	cerr.rdbuf( oss.rdbuf() );
	pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
	if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal) ) {
		ros::console::notifyLoggerLevelsChanged();
	}

	//
	// init and read parameter
	int a_numberOfSamples = 3;
	int a_correspondenceRandomness = 50;
	float a_similarityThreshold = 0.5;
	float a_maxCorrespondenceDistanceMultiplier = 0.3;
	float a_inlierFraction = 0.05;
	int a_maximumIterations = 50000;
	float vg_leafSize = 0.01;
	float nest_radius = 0.01;
	float fest_radius = 0.025;

	ros::param::get("~a_numberOfSamples", a_numberOfSamples);
	ros::param::get("~a_correspondenceRandomness", a_correspondenceRandomness);
	ros::param::get("~a_similarityThreshold", a_similarityThreshold);
	ros::param::get("~a_maxCorrespondenceDistanceMultiplier", a_maxCorrespondenceDistanceMultiplier);
	ros::param::get("~a_inlierFraction", a_inlierFraction);
	ros::param::get("~a_maximumIterations", a_maximumIterations);
	ros::param::get("~vg_leafSize", vg_leafSize);
	ros::param::get("~nest_radius", nest_radius);
	ros::param::get("~fest_radius", fest_radius);


	sensor_msgs::CameraInfo camera_info_msg;
	camera_info_msg.K = { {570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0} };
	string model_file = "/home/stfn/testdata/can_1_cloud.pcd";
	ros::param::get("~model_file", model_file);
 

	//
	// init aligner with parameters
	GODAlignment aligner(camera_info_msg.K);
	aligner.align.setNumberOfSamples(a_numberOfSamples);
	aligner.align.setCorrespondenceRandomness(a_correspondenceRandomness);
	aligner.align.setSimilarityThreshold(a_similarityThreshold);
	aligner.align.setMaxCorrespondenceDistance(a_maxCorrespondenceDistanceMultiplier * vg_leafSize);
	aligner.align.setInlierFraction(a_inlierFraction);
	aligner.align.setMaximumIterations(a_maximumIterations);
	aligner.vg.setLeafSize(vg_leafSize, vg_leafSize, vg_leafSize);
	aligner.nest.setRadiusSearch(nest_radius);
	aligner.fest.setRadiusSearch(fest_radius);
	aligner.loadModel(model_file);

#ifdef DEBUG
	pcl::visualization::PCLVisualizer visu;
#endif


	//
	// read bag into data container
	vector<geometry_msgs::PoseStamped> vec_pose;
	vector<cv::Mat> vec_rgb;
	vector<cv::Mat> vec_depth;
	vector<cv::Mat> vec_mask;

	rosbag::Bag bag;
	bag.open("/home/stfn/testdata/bags/can_1.bag", rosbag::bagmode::Read);

	vector<string> topics;
	topics.push_back(string("/camera/pose"));
	topics.push_back(string("/camera/rgb/image_color"));
	topics.push_back(string("/camera/depth/image"));
	topics.push_back(string("/camera/mask"));
	rosbag::View view(bag, rosbag::TopicQuery(topics));

	BOOST_FOREACH(rosbag::MessageInstance const m, view) {
		if (m.getTopic() == topics[0]) {
			geometry_msgs::PoseStamped::ConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();
			vec_pose.push_back(*pose);
		} else if (m.getTopic() == topics[1]) {
			cv_bridge::CvImagePtr cv_ptr;
			sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
			cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
			vec_rgb.push_back(cv_ptr->image);
		} else if (m.getTopic() == topics[2]) {
			cv_bridge::CvImagePtr cv_ptr;
			sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
			cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
			vec_depth.push_back(cv_ptr->image);
		} else if (m.getTopic() == topics[3]) {
			cv_bridge::CvImagePtr cv_ptr;
			sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
			cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
			vec_mask.push_back(cv_ptr->image);
		}
	}

	bag.close();


	//
	// do the alignment
	float error = 0;
	int skip = (vec_pose.size() >= 10)? vec_pose.size()/10 : 1;
	for (int i = 0; i < vec_pose.size(); i+=skip) {
		Mat &rgb_img = vec_rgb[i];
		Mat &depth_img = vec_depth[i];
		depth_img.convertTo(depth_img, CV_32FC1);
		Mat &mask_img = vec_mask[i];
		geometry_msgs::PoseStamped &p = vec_pose[i];

		Eigen::Translation3f trans(p.pose.position.x, p.pose.position.y, p.pose.position.z);
		Eigen::Quaternionf rot(p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z);
		Eigen::Affine3f pose_groundtruth = trans * rot;

		// create pointcloud from images
		PointCloudT::Ptr cloud(new PointCloudT);
		calcPointsRGBPCL(depth_img, rgb_img, cloud, 1.0);

		/*
		Eigen::Affine3f t_model_init = Eigen::Affine3f::Identity();
		t_model_init.rotate(Eigen::AngleAxisf (M_PI / 2, Eigen::Vector3f::UnitX()));
		pcl::transformPointCloud(*model, *model, t_model_init);
		pose_groundtruth.rotate(t_model_init.rotation().inverse());
		*/




		// generate center
		PointCloudT::Ptr cluster_cloud(new PointCloudT());
		Eigen::Vector2f center_of_mass(0, 0);
		int count = 0;
		for (int y = 0; y < mask_img.rows; ++y) {
			for (int x = 0; x < mask_img.cols; ++x) {
				if (mask_img.ptr<uchar>(y)[x]) {
					center_of_mass += Eigen::Vector2f(x, y);
					count++;
				}
			}
		}
		center_of_mass /= count;
		aligner.setInputCloud(cloud);
		pcl::PointXYZ cluster_center = aligner.extract_hypothesis_cluster_radius(cluster_cloud, center_of_mass[0], center_of_mass[1]);


		// align
		PointCloudT::Ptr model_aligned(new PointCloudT);
		Eigen::Matrix4f pose_prediction_mat;
		Eigen::Affine3f pose_prediction;
		pcl::StopWatch t1;
		bool success = aligner.align_cloud_to_model(cluster_cloud, pose_prediction_mat, model_aligned);

		if (success) {
			pose_prediction = Eigen::Affine3f(pose_prediction_mat);
			//pose_prediction.rotate(t_model_init.rotation().inverse());

			float off_trans = (pose_prediction.translation() - pose_groundtruth.translation()).norm();
			Eigen::AngleAxisf off_rot(pose_groundtruth.rotation().inverse()*pose_prediction.rotation());
			Eigen::Vector3f off_rot_err = (off_rot.axis() * off_rot.angle()).cwiseAbs();
			// create error
			error += 5 * off_trans + off_rot_err[0] + off_rot_err[1] + off_rot_err[2] / 10.f + t1.getTimeSeconds() / 2;
		} else {
			error += 50.0 + t1.getTimeSeconds() / 2;
		}


#ifdef DEBUG
		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/stfn/testdata/can_1_cloud.pcd", *model) < 0) {
			ROS_ERROR("Error loading model object file!");
			return -1;
		}
		visu.removeAllShapes();
		visu.removeAllPointClouds();
		cloud->is_dense = false;
		visu.addPointCloud<PointT>(cloud, pcl::visualization::PointCloudColorHandlerRGBField<PointT>(cloud), "scene");
		visu.addPointCloud<pcl::PointXYZ>(model, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(model, 255, 0, 0), "model");
		
		visu.addCoordinateSystem(0.1);
		visu.addCoordinateSystem(0.15, pose_groundtruth);
		visu.addCoordinateSystem(0.15, pose_prediction);

		visu.addPointCloud<PointT>(cluster_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(cluster_cloud, 0, 255, 0), "cluster");
		visu.addPointCloud<PointT>(model_aligned,  pcl::visualization::PointCloudColorHandlerCustom<PointT>(model_aligned, 255, 0, 0), "model_aligned");
		visu.spin(); 
#endif
	}


	cout.rdbuf( oldCoutStreamBuf );
	printf("error:%.5f\n", error);

	return 0;
}