#include "ModelDBStub.hpp"

#include <opencv2/core/core.hpp>

#define PCL_NO_PRECOMPILE
#include <pcl/octree/octree.h>
#include <pcl/octree/impl/octree_search.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include <tf/transform_listener.h>


#include <memory>

using namespace std;

/**
 *	TODO:	* OpenCV Abhängigkeit?
 *
 *
**/

struct DetectionClusterPoint {
	PCL_ADD_POINT4D; 
	float confidence;
	int class_id;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (DetectionClusterPoint,
	(float, x, x)
	(float, y, y)
	(float, z, z)
	(float, confidence, confidence)
	(int, class_id, class_id)
)
//PCL_INSTANTIATE(OctreePointCloudSearch, DetectionClusterPoint)
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;



class GODMapping {
public:
	typedef shared_ptr<GODMapping> Ptr;
	typedef shared_ptr<GODMapping const> ConstPtr;
	GODMapping();
	~GODMapping();
	void update(cv::Mat &depth_img, PointCloudT::Ptr &cloud, vector<vector<float> > &candidates, vector<vector<cv::Point2f> > &boundingboxes);
	void merge_clusters(DetectionClusterPoint &query_cluster, DetectionClusterPoint &persistent_cluster);
	void add_cluster(DetectionClusterPoint &query_cluster);
	void lookup_cam_transform(ros::Time &t, Eigen::Affine3d eigen_transform);

private:
	pcl::octree::OctreePointCloudSearch<DetectionClusterPoint>::Ptr octree;
	pcl::PointCloud<DetectionClusterPoint>::Ptr detection_cloud;
	shared_ptr<pcl::visualization::PCLVisualizer> visu;
	tf::TransformListener listener;
	ModelDBStub model_db;
};