#pragma once

#include "GODDetectionParams.hpp"
#include "CRForestDetector.hpp"
#include "CRForest.hpp"

#include <ros/ros.h>

#include <opencv2/core/core.hpp>



class GODDetection {
public:
	typedef shared_ptr<GODDetection> Ptr;
	typedef shared_ptr<GODDetection const> ConstPtr;
	GODDetection();
	~GODDetection();
	void initDetector();
	void detect(cv::Mat &rgb_img, cv::Mat &depth_img, vector<Candidate> &candidates);

private:
	GODDetectionParams::Ptr params;
	CRForest::Ptr crForest;
	CRForestDetector::Ptr crDetect;
};