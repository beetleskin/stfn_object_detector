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


#ifndef DEBUG
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
#endif	

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
	int runs = 0;
	int skip = (vec_pose.size() >= 10)? vec_pose.size()/10 : 1;
	for (int i = 0; i < vec_pose.size(); i+=skip, ++runs) {
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

		float error_alignment = 0;
		float error_time = 0;
		if (success) {
			pose_prediction = Eigen::Affine3f(pose_prediction_mat);
			// calculate translational error
			float off_trans = (pose_prediction.translation() - pose_groundtruth.translation()).norm();
			// calculate rotational error
			// rotaion-matrix == unit vectors! das hier drüber is also überflüssig ... einfach die spalten auslesen. hui.
			// ggf. noch mehr metriken also diese 3 abstände?
			float off_rot_x = (pose_prediction * Eigen::Vector4f::UnitX()- pose_groundtruth * Eigen::Vector4f::UnitX()).norm();
			float off_rot_y = (pose_prediction * Eigen::Vector4f::UnitY()- pose_groundtruth * Eigen::Vector4f::UnitY()).norm();
			float off_rot_z = (pose_prediction * Eigen::Vector4f::UnitZ()- pose_groundtruth * Eigen::Vector4f::UnitZ()).norm();
			
			error_alignment = off_trans*100 + off_rot_x/10 + off_rot_y/10 + off_rot_z*2;
#ifdef DEBUG
			cout << "err trans: " << off_trans << endl;
			cout << "err rot: " << off_rot_x << " " << off_rot_y << " " << off_rot_z << " " << endl;
#endif

		} else {
			error_alignment = 50;
		}


		error_time = t1.getTimeSeconds();
		error += error_alignment + error_time/2;

#ifdef DEBUG
		cout << "err time: " << error_time << endl;
		cout << "err total: " << error << endl;
		cout << endl;
#endif


#ifdef DEBUG
		// load model
		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/stfn/testdata/can_1_cloud.pcd", *model) < 0) {
			ROS_ERROR("Error loading model object file!");
			return -1;
		}

		// clean scene
		visu.removeAllShapes();
		visu.removeAllPointClouds();
		visu.removeCoordinateSystem("OO");
		visu.removeCoordinateSystem("pose_groundtruth");
		visu.removeCoordinateSystem("pose_prediction");


		// add stuff
		cloud->is_dense = false;
		visu.addPointCloud<PointT>(cloud, pcl::visualization::PointCloudColorHandlerRGBField<PointT>(cloud), "scene");
		visu.addPointCloud<pcl::PointXYZ>(model, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(model, 255, 0, 0), "model");
		

		visu.addCoordinateSystem(0.1, "OO");
		visu.addCoordinateSystem(0.15, pose_groundtruth, "pose_groundtruth");
		visu.addCoordinateSystem(0.15, pose_prediction, "pose_prediction");

		visu.addPointCloud<PointT>(cluster_cloud, pcl::visualization::PointCloudColorHandlerCustom<PointT>(cluster_cloud, 0, 255, 0), "cluster");
		visu.addPointCloud<PointT>(model_aligned,  pcl::visualization::PointCloudColorHandlerCustom<PointT>(model_aligned, 255, 0, 0), "model_aligned");
		visu.spin(); 
#endif

		if(error_time > 2)
			break;
	}

#ifndef DEBUG
	// reset output buffer to default
	cout.rdbuf( oldCoutStreamBuf );
#endif

	error /= runs;
	if(!isfinite(error))
		error = 50.f;
	printf("error:%.10f\n", error);
	return 0;
}
