#include <stfn_object_detector/Detection2D.h>
#include <stfn_object_detector/Detection2DArr.h>

#include <pcl/point_types.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>


using namespace std;


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, stfn_object_detector::Detection2DArr> MySyncPolicy;


struct ModelData {
	int id;
	float detection_threshold;
	float consideration_threshold;
};
vector<ModelData> models;



void init_stub_data() {
	models.resize(1);
	ModelData model_soda_can;
	model_soda_can.id = 0;
	model_soda_can.detection_threshold = 5;
	model_soda_can.consideration_threshold = 1.0;
	models[0] = model_soda_can;
}


void det_pcl_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const stfn_object_detector::Detection2DArrConstPtr &det_msg) {
	pcl::PCLPointCloud2 pcl_pc;
	PointCloudT::Ptr cloud(new PointCloudT);
	pcl_conversions::toPCL(*cloud_msg, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, *cloud);



	for (int i = 0; i < det_msg->detections.size(); ++i) {
		const stfn_object_detector::Detection2D &d = det_msg->detections[i];
		ROS_INFO("got detection: [%.1f, %.1f](%.1f, %.1f)", d.leftTop.x, d.leftTop.y, d.bottomRight.x - d.leftTop.x, d.bottomRight.y - d.leftTop.y);
	}
}


int main(int argc, char **argv) {
	ros::init(argc, argv, "mapper");

	ros::NodeHandle nh;

	init_stub_data();

	
	// register synced subscribers
	message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "tmp_cloud", 1);
	message_filters::Subscriber<stfn_object_detector::Detection2DArr> det_sub(nh, "dhf_detections", 1);
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pcl_sub, det_sub);
	sync.registerCallback(bind(&det_pcl_callback, _1, _2 ));


	ros::spin();
	return 0;
}