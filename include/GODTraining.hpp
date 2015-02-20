#include "CRForestDetector.hpp"
#include "CRForest.hpp"

#include <ros/ros.h>

#include <opencv2/core/core.hpp>

using namespace std;



class GODTraining {
public:
	typedef shared_ptr<GODTraining> Ptr;
	typedef shared_ptr<GODTraining const> ConstPtr;
	GODTraining();
	~GODTraining();
	void initTraining();
	void train();
	void load_traindata(CRPatch &Train, CvRNG *pRNG);

private:
	CRForest::Ptr crForest;
	CRForestDetector::Ptr crDetect;
	float initial_scale;
	float nlabels;
};