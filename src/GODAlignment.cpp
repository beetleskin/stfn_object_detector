#include "GODAlignment.hpp"

#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

#include <ros/ros.h>


using namespace std;



GODAlignment::GODAlignment(boost::array<double, 9ul> K) :
	caminfo_K(K) {

	//
	// setup pcl pipeline
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	nest.setSearchMethod(tree);

	this->initParams();

	this->scene.reset(new PointCloudT);
}


void GODAlignment::initParams() {


	//
	// init and read parameter

	float vg_leafSize = 0.01;
	int numberOfSamples = 3;
	int correspondenceRandomness = 50;
	float similarityThreshold = 0.5;
	float maxCorrespondenceDistanceMultiplier = 0.3;
	float inlierFraction = 0.05;
	int maximumIterations = 50000;
	float nest_radius = 0.01;
	float fest_radius = 0.025;


	ros::param::get("~aligner_leafSize", vg_leafSize);
	ros::param::get("~aligner_numberOfSamples", numberOfSamples);
	ros::param::get("~aligner_correspondenceRandomness", correspondenceRandomness);
	ros::param::get("~aligner_similarityThreshold", similarityThreshold);
	ros::param::get("~aligner_maxCorrespondenceDistanceMultiplier", maxCorrespondenceDistanceMultiplier);
	ros::param::get("~aligner_inlierFraction", inlierFraction);
	ros::param::get("~aligner_maximumIterations", maximumIterations);
	ros::param::get("~aligner_nest_radius", nest_radius);
	ros::param::get("~aligner_fest_radius", fest_radius);



	//
	// set ros dynamic parameter

	vg.setLeafSize(vg_leafSize, vg_leafSize, vg_leafSize);

	// Number of points to sample for generating/prerejecting a pose
	align.setNumberOfSamples(numberOfSamples);
	// Number of nearest features to use
	align.setCorrespondenceRandomness(correspondenceRandomness);
	// Polygonal edge length similarity threshold
	align.setSimilarityThreshold(similarityThreshold);
	// Set inlier threshold
	align.setMaxCorrespondenceDistance(maxCorrespondenceDistanceMultiplier * vg.getLeafSize()[0]);
	// Set required inlier fraction
	align.setInlierFraction(inlierFraction);
	// Set number of maximum iterations
	align.setMaximumIterations(maximumIterations);

	nest.setRadiusSearch(nest_radius);

	fest.setRadiusSearch(fest_radius);

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.02);
}


void GODAlignment::setInputCloud(PointCloudT::Ptr inputCloud) {
	this->scene = inputCloud;
}


bool GODAlignment::align_cloud_to_model(PointCloudT::Ptr cluster, Eigen::Matrix4f &transformation, PointCloudT::Ptr model_aligned) {
	// @see http://pointclouds.org/documentation/tutorials/alignment_prerejective.php#alignment-prerejective
	pcl::StopWatch t1;

	// resample to match model point density
	vg.setInputCloud(cluster);
	vg.filter(*cluster);

	// compute normals
	NormalCloudT::Ptr cluster_normals (new NormalCloudT);
	nest.setInputCloud(cluster);
	nest.compute(*cluster_normals);

	// estimate features
	FeatureCloudT::Ptr cluster_features(new FeatureCloudT);
	fest.setInputCloud(cluster);
	fest.setInputNormals(cluster_normals);
	fest.compute(*cluster_features);

	ROS_DEBUG("\tfound %i features", (int) cluster_features->size());
	ROS_DEBUG("\tstarting alignment");

	align.setInputSource(this->model);
	align.setSourceFeatures(this->model_features);
	align.setInputTarget(cluster);
	align.setTargetFeatures(cluster_features);

	align.align(*model_aligned);
	// TODO: followed by icp?


	// check alignment
	if (align.hasConverged()) {

		transformation = align.getFinalTransformation();

		ROS_INFO("\tAlignment successful.");
		ROS_INFO("\t    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1), transformation(0, 2));
		ROS_INFO("\tR = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1), transformation(1, 2));
		ROS_INFO("\t    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1), transformation(2, 2));
		ROS_INFO("\n");
		ROS_INFO("\tt = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3), transformation(2, 3));
		ROS_INFO("\n");
		ROS_INFO("\tInliers: %i/%i = %f\n", align.getInliers().size(), this->model->size(), (float)align.getInliers().size() / this->model->size() * 100);
		ROS_INFO("\tScore: %i/%i = %f\n", align.getInliers().size(), this->model->size(), (float)align.getInliers().size() / this->model->size() * 100);

		return true;

	} else {
		ROS_INFO("\tAlignment failed!");
		return false;
	}

	ROS_INFO("\ttime consumed: %0.3fs", t1.getTimeSeconds());
}



void GODAlignment::extract_hypothesis_cluster_crop(PointCloudT::Ptr &coud_clip, int off_x, int off_y, int width, int height) {
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	off_x -= width / 2;
	off_y -= height / 2;
	width *= 2;
	height *= 2;
	if (off_x < 0)
		off_x = 0;
	if (off_y < 0)
		off_y = 0;
	if (off_x + width >= this->scene->width)
		width = this->scene->width - off_x - 1;
	if (off_y + height >= this->scene->height)
		height = this->scene->height - off_y - 1;

	for (int y = off_y; y < off_y + height; ++y) {
		for (int x = off_x; x < off_x + width; ++x) {
			int i = y * this->scene->width + x;
			if (pcl::isFinite (this->scene->points[i]))
				inliers->indices.push_back(i);
		}
	}

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(this->scene);
	extract.setIndices (inliers);
	extract.setNegative (false);
	extract.filter (*coud_clip);
}


pcl::PointXYZ GODAlignment::extract_hypothesis_cluster_radius(PointCloudT::Ptr &cluster_cloud, float x, float y) {
	// prepare radius search
	pcl::search::KdTree<PointT>::Ptr kd_tree (new pcl::search::KdTree<PointT> ());
	kd_tree->setInputCloud(this->scene);

	// find 3D cluster center
	PointT cluster_center = this->scene->at(x, y);
	ROS_DEBUG("\t(x,y,z)=(%f,%f, %f)", cluster_center.x, cluster_center.y, cluster_center.z);

	// extract bounding sphere
	boost::shared_ptr<std::vector<int> > inlier_indices (new std::vector<int>());
	boost::shared_ptr<std::vector<float> > inlier_radii (new std::vector<float>());
	kd_tree->radiusSearch(cluster_center, this->model_bounding_sphere_r * 1.5, *inlier_indices, *inlier_radii);
	pcl::ExtractIndices<PointT> extract_indices;
	extract_indices.setInputCloud(this->scene);
	extract_indices.setIndices(inlier_indices);
	extract_indices.filter(*cluster_cloud);

	return pcl::PointXYZ(cluster_center.x, cluster_center.y, cluster_center.z);
}



void GODAlignment::remove_planes(PointCloudT::Ptr &cloud) {
	// compute normals
	NormalCloudT::Ptr cloud_normals (new NormalCloudT);
	nest.setInputCloud (cloud);
	nest.compute (*cloud_normals);

	// organized multi plane segmentation
	std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
	std::vector<pcl::ModelCoefficients> model_coefficients;
	std::vector<pcl::PointIndices> inlier_indices;
	pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>());
	std::vector<pcl::PointIndices> label_indices;
	std::vector<pcl::PointIndices> boundary_indices;
	pcl::OrganizedMultiPlaneSegmentation<PointT, NormalT, pcl::Label> multiPlane;
	multiPlane.setMinInliers(5000);
	multiPlane.setDistanceThreshold(0.01);
	multiPlane.setAngularThreshold(pcl::deg2rad(5.0));
	// multiPlane.setMaximumCurvature(...)
	multiPlane.setInputCloud(cloud);
	multiPlane.setInputNormals(cloud_normals);
	multiPlane.segmentAndRefine(regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

	ROS_DEBUG("Found %i planes", (int)inlier_indices.size());

	// merge inliers
	int inliers_total = 0;
	boost::shared_ptr<std::vector<int> > inliers (new std::vector<int>());
	for (int i = 0; i < inlier_indices.size(); ++i) {
		inliers->insert(inliers->end(), inlier_indices[i].indices.begin(), inlier_indices[i].indices.end());
		inliers_total += inlier_indices[i].indices.size();
	}

	// remove planes (and show them)
	pcl::ExtractIndices<PointT> extract_indices;
	extract_indices.setInputCloud(cloud);
	extract_indices.setIndices(inliers);
	// PointCloudT::Ptr planes(new PointCloudT());
	// extract_indices.setNegative(false);
	// extract_indices.filter(*planes);
	// visu.addPointCloud<PointT>(planes, pcl::visualization::PointCloudColorHandlerCustom<PointT>(planes, 255, 0, 255), "planes");
	// visu.spinOnce();
	extract_indices.setNegative(true);
	extract_indices.filter(*cloud);
}


void GODAlignment::loadModel(std::string model_file) {
	this->model = PointCloudT::Ptr(new PointCloudT);
	PointCloudT::Ptr model_original(new PointCloudT());

	if (pcl::io::loadPCDFile<PointT>(model_file, *model_original) < 0) {
		ROS_ERROR("Error loading model object file!");
		// TODO: throw exception
	}

	// voxelfilter it
	vg.setInputCloud(model_original);
	vg.filter(*this->model);

	// get AABB
	PointT model_bb_min_p, model_bb_max_p;
	pcl::getMinMax3D(*this->model, model_bb_min_p, model_bb_max_p);
	this->model_bounding_sphere_r = (model_bb_max_p.x - model_bb_min_p.x) * (model_bb_max_p.x - model_bb_min_p.x) + (model_bb_max_p.y - model_bb_min_p.y) * (model_bb_max_p.y - model_bb_min_p.y) + (model_bb_max_p.z - model_bb_min_p.z) * (model_bb_max_p.z - model_bb_min_p.z);
	this->model_bounding_sphere_r = sqrt(this->model_bounding_sphere_r) / 2;

	// compute normals
	NormalCloudT::Ptr model_normals(new NormalCloudT());
	nest.setInputCloud(this->model);
	nest.compute(*model_normals);

	// estimate features
	this->model_features = FeatureCloudT::Ptr(new FeatureCloudT);
	fest.setInputCloud(this->model);
	fest.setInputNormals(model_normals);
	fest.compute(*this->model_features);

	ROS_DEBUG("Model loaded");
	ROS_DEBUG("\t found %i features", (int) this->model_features->size());
}


GODAlignment::~GODAlignment() {
}
