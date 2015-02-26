#include "GODDetection.hpp"
#include "GODImageMemory.hpp"

#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#include <opencv2/highgui/highgui.hpp>



GODDetection::GODDetection() {
	// set number of cores for multiprocessing
	tbb::task_scheduler_init init(5);

	// initialize the detector with ros parameters
	this->params = GODDetectionParams::get();
	initDetector();
}


GODDetection::~GODDetection() {
}


void GODDetection::initDetector() {

	// init forest
	crForest.reset(new CRForest(params->number_of_trees));
	crForest->loadForest(params->tree_path.c_str(), 0);
	params->nlabels = crForest->GetNumLabels();

	// init detector
	crDetect.reset(new CRForestDetector(crForest));
}


void GODDetection::detect(cv::Mat &rgb_img, cv::Mat &depth_img, vector<Candidate> &candidates) {
	tbb::task_group tbb_tg;
	Mat rgb_small, depth_small;

	// resize images to downscale size
	resize(rgb_img, rgb_small, Size(), params->image_scale, params->image_scale, CV_INTER_LINEAR);
	resize(depth_img, depth_small, Size(), params->image_scale, params->image_scale, CV_INTER_NN);


	//
	// classify each pixel and assign leaf node IDs
	crDetect->assignCluster(rgb_small, depth_small);


	//
	// calculate the class priors for each pixel
	crDetect->calcClassPriors();


	//
	// detect (multithread)
	vector<vector<Candidate> > temp_candidates(params->nlabels - 1);
	for (unsigned int cNr = 0; cNr < params->nlabels - 1; cNr++) {
		float threshold_this_class = params->threshold_hierarchy / float(params->nlabels);
		auto job_func = bind(&CRForestDetector::detectPyramidMR, crDetect, ref(temp_candidates[cNr]), ref(depth_small), cNr, threshold_this_class);
		tbb_tg.run(job_func);
	}
	tbb_tg.wait();


	//
	// finalize candidates
	candidates.clear();
	for (unsigned int cNr = 0; cNr < params->nlabels - 1; cNr++) {
		for (unsigned int candNr = 0; candNr < temp_candidates[cNr].size(); candNr++) {
			candidates.push_back(temp_candidates[cNr][candNr]);
		}
	}

	sort(candidates.begin(), candidates.end(), 
		[](const Candidate &a, const Candidate &b) {
			return a.confidence > b.confidence;
	});

	if(params->do_backprojection) {
		for (size_t candNr = 0; candNr < candidates.size(); candNr++) {
			auto job_func = [ &, candNr]() {
				Candidate &cand = candidates[candNr];
				crDetect->voteForCandidate(depth_small, cand);
				crDetect->backprojectCandidate(rgb_small, cand);
			};
			tbb_tg.run(job_func);
		}
		tbb_tg.wait();
	}
	
	
	vector<string> model_names = params->models;
	// compensate for downscale and assign external class name
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {

		Candidate &cand = candidates[candNr];
		cand.class_name = model_names[cand.class_id];
		
		cand.center.x /= params->image_scale;
		cand.center.y /= params->image_scale;
		cand.bb[0].x /= params->image_scale;
		cand.bb[0].y /= params->image_scale;
		cand.bb[1].x /= params->image_scale;
		cand.bb[1].y /= params->image_scale;
	}













#if 1
	// show the hough space
	for (int i = 0; i < params->nlabels-1; i++) {
		cv::Mat hough = GODImageMemory::hough_space[i].clone();
		cv::normalize(hough, hough, 0, 1.0, cv::NORM_MINMAX);
		string window_name = "hough [" + model_names[i] + "]";
		cv::imshow(window_name, hough);
	}


	// draw the candidates
	for (size_t candNr = 0; candNr < candidates.size(); candNr++) {

		Candidate &cand = candidates[candNr];
		rectangle(rgb_img, cand.bb[0], cand.bb[1], Scalar(cand.class_id*255, 0, 255), 3);

	}
	cv::imshow("detections", rgb_img);



	cv::waitKey(10);
#endif


}