#include "GODDetection.hpp"

#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>



GODDetection::GODDetection() {
	// set number of cores for multiprocessing
	tbb::task_scheduler_init init(5);

	// initialize the detector with ros parameters
	initDetector();
}

GODDetection::~GODDetection() {
}


void GODDetection::initDetector() {
	// default values for params
	kernel_sizes.resize(3, 20);
	scales.resize(1, 1.0);
	threshold_hierarchy = 1.5;
	max_candidates = 20;
	nlabels = -1;
	detection_threshold = 0.1;
	do_backprojection = true;
	cand_max_width = 320;
	cand_max_height = 240;

	// temp params
	std::string forest_path;
	int num_trees = 15;
	int off_trees = 0;
	int p_width = 10;
	int p_height = 10;

	// override params from ros parameter server
	ros::param::get("~forest_path", forest_path);
	ros::param::get("~num_trees", num_trees);
	ros::param::get("~off_trees", off_trees);
	ros::param::get("~p_width", p_width);
	ros::param::get("~p_height", p_height);
	ros::param::get("~initial_scale", initial_scale);
	ros::param::get("~kernel_smooth", kernel_sizes[0]);
	ros::param::get("~kernel_vote", kernel_sizes[1]);
	ros::param::get("~kernel_sigma", kernel_sizes[2]);
	ros::param::get("~threshold_hierarchy", threshold_hierarchy);
	ros::param::get("~max_candidates", max_candidates);
	ros::param::get("~detection_threshold", detection_threshold);
	ros::param::get("~do_backprojection", do_backprojection);
	ros::param::get("~cand_max_width", cand_max_width);
	ros::param::get("~cand_max_height", cand_max_height);

	// init forest
	crForest.reset(new CRForest(num_trees));
	crForest->loadForest(forest_path.c_str(), off_trees);
	nlabels = crForest->GetNumLabels();
	vector<int> temp_classes;
	temp_classes.resize(1);
	temp_classes[0] = -1;
	crForest->SetTrainingLabelsForDetection(temp_classes);

	// init detector
	crDetect.reset(new CRForestDetector(crForest, p_width, p_height));
}


void GODDetection::detect(cv::Mat &rgb_img, cv::Mat &depth_img, vector<vector<float> > &candidates, vector<vector<cv::Point2f> > &boundingboxes) {
	tbb::task_group tbb_tg;
	Mat rgb_small, depth_small;

	// resize images to downscale size
	resize(rgb_img, rgb_small, Size(), initial_scale, initial_scale, CV_INTER_LINEAR);
	resize(depth_img, depth_small, Size(), initial_scale, initial_scale, CV_INTER_NN);


	//
	// assign cluster
	vector<vector<Mat> > vImgAssign;
	crDetect->fullAssignCluster(rgb_small, depth_small, vImgAssign, scales);


	//
	// get class confidence
	vector<vector<Mat> > classConfidence;
	crDetect->getClassConfidence(vImgAssign, classConfidence);


	//
	// detect (multithread)
	vector<vector<vector<float > > > temp_candidates(nlabels - 1);
	for (unsigned int cNr = 0; cNr < nlabels - 1; cNr++) {
		float threshold_this_class = threshold_hierarchy / float(nlabels);
		vector<float> params(4);
		params[0] = max_candidates;
		params[1] = cNr;
		params[2] = detection_threshold;
		params[3] = threshold_this_class;
		auto job_func = bind(&CRForestDetector::detectPyramidMR, crDetect, ref(vImgAssign), ref(temp_candidates[cNr]), scales, kernel_sizes, params, ref(classConfidence), ref(depth_small));
		tbb_tg.run(job_func);
	}
	tbb_tg.wait();


	//
	// finalize candidates and boundingboxes
	for (unsigned int cNr = 0; cNr < nlabels - 1; cNr++) {
		for (unsigned int candNr = 0; candNr < temp_candidates[cNr].size(); candNr++) {
			candidates.push_back(temp_candidates[cNr][candNr]);
		}
	}

	sort(candidates.begin(), candidates.end(), 
		[](const vector<float> &a, const vector<float> &b) {
			return a[0] > b[0];
	});
	boundingboxes.resize(candidates.size());
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {
		int scNr = 0;
		auto job_func = [ &, candNr]() {
			Candidate cand(crForest, rgb_small, candidates[candNr], candNr, do_backprojection);
			crDetect->voteForCandidate(depth_small, vImgAssign[scNr], cand, kernel_sizes[0], cand_max_width, cand_max_height);
			cand.getBBfromBpr();
			boundingboxes[candNr] = cand.bb_;
		};
		tbb_tg.run(job_func);
	}
	tbb_tg.wait();

	// compensate for downscale
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {
		candidates[candNr][1] /= initial_scale;
		candidates[candNr][2] /= initial_scale;
		boundingboxes[candNr][0].x /= initial_scale;
		boundingboxes[candNr][0].y /= initial_scale;
		boundingboxes[candNr][1].x /= initial_scale;
		boundingboxes[candNr][1].y /= initial_scale;
	}
}
