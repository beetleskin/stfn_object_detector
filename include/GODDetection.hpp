#include "CRForestDetector.hpp"
#include "CRForest.hpp"

#include <ros/ros.h>

#include <opencv2/core/core.hpp>

using namespace std;



class GODDetection {
public:
	typedef shared_ptr<GODDetection> Ptr;
	typedef shared_ptr<GODDetection const> ConstPtr;
	GODDetection();
	~GODDetection();
	void initDetector();
	void detect(cv::Mat &rgb_img, cv::Mat &depth_img, vector<vector<float> > &candidates, vector<vector<cv::Point2f> > &boundingboxes);

private:
	CRForest::Ptr crForest;
	CRForestDetector::Ptr crDetect;
	float initial_scale;
	vector<float> kernel_sizes;
	vector<float> scales;
	float threshold_hierarchy;
	float max_candidates;
	float nlabels;
	float detection_threshold;
	bool do_backprojection;
	float cand_max_width;
	float cand_max_height;
};