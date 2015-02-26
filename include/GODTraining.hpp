#pragma once

#include "GODDetectionParams.hpp"
#include "CRForest.hpp"

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



class GODTraining {
public:
	typedef shared_ptr<GODTraining> Ptr;
	typedef shared_ptr<GODTraining const> ConstPtr;
	GODTraining();
	~GODTraining();
	void initTraining();
	void train();
	std::map<std::string, int> load_traindata(CRPatch &Train, CvRNG *pRNG);

private:
	GODDetectionParams::Ptr params;
	CRForest::Ptr forest_ptr;
};