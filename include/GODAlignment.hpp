#include <boost/array.hpp>

#include <pcl/common/time.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>



typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<NormalT> NormalCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointT, NormalT, FeatureT> FeatureEstimationT;
//typedef pcl::PFHRGBSignature250 FeatureT;
//typedef pcl::PFHRGBEstimation<PointT, PointT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;


using namespace std;


class GODAlignment {
public:
	typedef shared_ptr<GODAlignment> Ptr;
	typedef shared_ptr<GODAlignment const> ConstPtr;
	GODAlignment(boost::array<double, 9ul> K);
	~GODAlignment();
	bool align_cloud_to_model(PointCloudT::Ptr cluster, Eigen::Matrix4f &transformation, PointCloudT::Ptr model_aligned);
	void extract_hypothesis_cluster_crop(PointCloudT::Ptr &coud_clip, int off_x, int off_y, int width, int height);
	pcl::PointXYZ extract_hypothesis_cluster_radius(PointCloudT::Ptr &cluster_cloud, float x, float y);
	void remove_planes(PointCloudT::Ptr &cloud);
	void loadModel(std::string model_file);

	void setInputCloud(PointCloudT::Ptr inputCloud);

public:
	PointCloudT::Ptr model;
	FeatureCloudT::Ptr model_features;
	PointCloudT::Ptr scene;
	pcl::VoxelGrid<PointT> vg;
	pcl::SACSegmentation<PointT> seg;
	FeatureEstimationT fest;
	pcl::NormalEstimationOMP<PointT, NormalT> nest;
	pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> align;

	boost::array<double, 9ul> caminfo_K;
	double model_bounding_sphere_r;

private:
	void initParams();
};