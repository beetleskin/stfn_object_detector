#include "GODMapping.hpp"
#include "myutils.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/common/transforms.h>

#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h>


GODMapping::GODMapping() {
	octree.reset(new pcl::octree::OctreePointCloudSearch<DetectionClusterPoint>(0.1));
	detection_cloud.reset(new pcl::PointCloud<DetectionClusterPoint>);
	octree->setInputCloud (detection_cloud);
	octree->addPointsFromInputCloud();

	visu.reset(new pcl::visualization::PCLVisualizer());
	visu->setBackgroundColor(0.7, 0.7, 0.7);
}


void GODMapping::lookup_cam_transform(ros::Time &t, Eigen::Affine3d eigen_transform) {
	try {
		tf::StampedTransform tf_transform;
		listener.lookupTransform("/camera_rgb_optical_frame", "/map", t, tf_transform);
		tf::transformTFToEigen (tf_transform, eigen_transform);
	} catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
    }
}


// TODO: cluster center is resolved in camera frame, make it relative to world frame
// TODO: parametrize model-radius for both, existing cluster and new ones
void GODMapping::update(cv::Mat &depth_img, PointCloudT::Ptr &cloud, vector<vector<float> > &candidates, vector<vector<cv::Point2f> > &boundingboxes) {
	

	// visu stuff
	visu->removeAllShapes();
	visu->removeAllPointClouds();


	// get the transform from camera to the map
	Eigen::Affine3d cam_map_tf;
	cam_map_tf.setIdentity();
	ros::Time t(0);
	lookup_cam_transform(t, cam_map_tf);


	// for each detection, add or merge to existing cluster map
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {
		const vector<float> &candidate = candidates[candNr];
		const vector<cv::Point2f> &bb = boundingboxes[candNr];

		//
		// check thresholds for individual classes
		if(candidate[0] < model_db.get_model_by_id(candidate[4]).threshold_consideration) {
			continue;
		}


		// TODO: sometimes this happens ... why?
		if(bb[0].x > bb[1].x || bb[0].y > bb[1].y) {
			ROS_WARN("Dropping invalid detection");
			continue;
		}


		//
		// delete detection from the depth image
		cv::Rect detection_rect(bb[0].x, bb[0].y, bb[1].x-bb[0].x, bb[1].y-bb[0].y);
		if(detection_rect.x < 0) detection_rect.x = 0;
		if(detection_rect.y < 0) detection_rect.y = 0;
		if(detection_rect.x+detection_rect.width >= depth_img.cols) detection_rect.width = depth_img.cols - detection_rect.x - 1;
		if(detection_rect.y+detection_rect.height >= depth_img.rows) detection_rect.height = depth_img.rows - detection_rect.y - 1;
		depth_img(detection_rect) = cv::Scalar(0);
		ROS_INFO("cand#%d: %f", (int)candNr, candidate[0]);


		//
		// get detection cluster center in 3D space
		PointT query_cluster_center(numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN());
		if(candidate[1] >= 0 && candidate[1] < cloud->width && candidate[2] >= 0 && candidate[2] < cloud->height)
			query_cluster_center = cloud->at(candidate[1], candidate[2]);
		if(!pcl::isFinite(query_cluster_center)) {
			ROS_ERROR("Detection cluster center is not finite! Aborting mapping for detection %d", (int)candNr);
			continue;
		}

		DetectionClusterPoint query_cluster;
		query_cluster.x = query_cluster_center.x;
		query_cluster.y = query_cluster_center.y;
		query_cluster.z = query_cluster_center.z;
		query_cluster.confidence = candidate[0];
		query_cluster.class_id = candidate[4];

		// transform it from cam coordinate system to global (map)
		pcl::transformPoint(query_cluster, cam_map_tf);

		
		//
		// find clusters near this cluster with radius search
		float search_radius = 0.1;
		vector<int> search_indices;
		vector<float> search_sqr_distances;
		if(octree->radiusSearch(query_cluster, search_radius, search_indices, search_sqr_distances) > 0) {
			
			
			for (int i = 0; i < search_indices.size(); ++i) {
				DetectionClusterPoint &matching_cluster = detection_cloud->points[search_indices[i]];
				// check radius again with correct params for existing and query cluster
				float query_cluster_class_radius = model_db.get_model_by_id(query_cluster.class_id).radius;
				float matching_cluster_class_radius = model_db.get_model_by_id(matching_cluster.class_id).radius;
				float combined_search_radius_sqr = (query_cluster_class_radius+matching_cluster_class_radius) * (query_cluster_class_radius+matching_cluster_class_radius);
				
				if(search_sqr_distances[i] < combined_search_radius_sqr) {
					if(query_cluster.class_id == matching_cluster.class_id) {
						// same class, merge clusters
						merge_clusters(query_cluster, matching_cluster);
						ROS_INFO("merging, new conf: %f", query_cluster.confidence);
					} else {
						// different class, what todo?
						// delete the old one, add the new one?
					}
				}

				// TODO: for now, only use first (aka nearest) match
				break;
			}
			
		} else {
			add_cluster(query_cluster);
			ROS_INFO("adding, new conf: %f", query_cluster.confidence);
		}
	}


	//
	// update cluster map by raytracing depth measures that do not contain detection clusters
	// downscale depth image
	cv::Mat clear_img;
	cv::resize(depth_img, clear_img, cv::Size(), 0.25, 0.25);
	//cv::imshow("clear_img", clear_img/1000);
	//cv::waitKey(50);

	// create pointcloud for raytracing
	pcl::PointCloud<pcl::PointXYZ>::Ptr clear_cloud;
	calcPointsPCL(clear_img, clear_cloud, 0.25);
	pcl::transformPointCloud(*clear_cloud, *clear_cloud, cam_map_tf);

	
	// raytrace every ray from the clear_cloud with the persistent detection clusters: delete intersections
	for (int i = 0; i < clear_cloud->points.size(); ++i) {
		const pcl::PointXYZ &p = clear_cloud->points[i];
		if(!pcl::isFinite(p) || p.z == 0)
			continue;

		// ray-sphere intersection: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
		Eigen::Vector3f x1(0,0,0);
		Eigen::Vector3f x2(clear_cloud->points[i].x, clear_cloud->points[i].y, clear_cloud->points[i].z);
		

		// TODO: use octree for fast raytracing
		for (int j = 0; j < detection_cloud->points.size(); ++j) {
			DetectionClusterPoint &cluster = detection_cloud->points[j];
			Eigen::Vector3f x0(cluster.x, cluster.y, cluster.z);

			float cluster_class_radius = 0.01;
			float ray_cluster_distance = (x0-x1).cross(x0-x2).norm() / (x2-x1).norm();
			if(ray_cluster_distance == ray_cluster_distance && ray_cluster_distance <= cluster_class_radius) {
				
				visu->addLine(pcl::PointXYZ(x1[0], x1[1], x1[2]), pcl::PointXYZ(x2[0], x2[1], x2[2]), 0, 0, 1, "line_" + std::to_string(i));
				
				// update cluster
				cluster.confidence *= 0.9;

				// delete cluster
				if(cluster.confidence < 0.1) {
					detection_cloud->points.erase(detection_cloud->points.begin() + j);
					j--;
				}
			}
		}
	}


	// OctreePointCloudSearch is not dynamic ... it has to be rebuilt
	octree->deleteTree();
	octree->addPointsFromInputCloud();


	//
	// visualize
	for (int i = 0; i < detection_cloud->points.size(); ++i) {
		const DetectionClusterPoint &c = detection_cloud->points[i];
		float cluster_class_radius = 0.05;
		float conf_max = 1.f;
		string id = "cluster_" + std::to_string(i);
		visu->addSphere<DetectionClusterPoint>(c, cluster_class_radius, (conf_max-c.confidence)/conf_max, (c.confidence)/conf_max, 0., id);
		visu->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
	}

	cloud->is_dense = false;
	pcl::transformPointCloud(*cloud, *cloud, cam_map_tf);
	visu->addPointCloud<PointT>(cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud), "scene");
	visu->spinOnce();




}


void GODMapping::merge_clusters(DetectionClusterPoint &query_cluster, DetectionClusterPoint &persistent_cluster) {
	float distance_to_sensor = sqrt(query_cluster.x*query_cluster.x + query_cluster.y*query_cluster.y + query_cluster.z*query_cluster.z);
	// weight detection confidence by distance
	query_cluster.confidence *= 1.f/(distance_to_sensor+0.3f);
	float confidence_sum = query_cluster.confidence + persistent_cluster.confidence;

	// TODO: better merge
	// use more points (history), filter? kalman? additionaly weight by time (regler?)
	persistent_cluster.x = (persistent_cluster.confidence*persistent_cluster.x + query_cluster.confidence*query_cluster.x) / confidence_sum;
	persistent_cluster.y = (persistent_cluster.confidence*persistent_cluster.y + query_cluster.confidence*query_cluster.y) / confidence_sum;
	persistent_cluster.z = (persistent_cluster.confidence*persistent_cluster.z + query_cluster.confidence*query_cluster.z) / confidence_sum;
	persistent_cluster.confidence = (persistent_cluster.confidence*persistent_cluster.confidence + query_cluster.confidence*query_cluster.confidence) / (persistent_cluster.confidence+query_cluster.confidence);
}


void GODMapping::add_cluster(DetectionClusterPoint &query_cluster) {
	float distance_to_sensor = sqrt(query_cluster.x*query_cluster.x + query_cluster.y*query_cluster.y + query_cluster.z*query_cluster.z);
	query_cluster.confidence *= 1.f/(distance_to_sensor+0.3f);
	octree->addPointToCloud(query_cluster, detection_cloud);
	ROS_INFO("#clusters: %d", int(detection_cloud->points.size()));
}


GODMapping::~GODMapping() {

}