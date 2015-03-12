#pragma once

#include "GODDetectionParams.hpp"
#include "CRForest.hpp"
#include "Candidate.hpp"

#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <vector>


using namespace std;
using namespace cv;

static vector<Mat> default_vImg__;
static Rect default_rect__;


class CRForestDetector {

private:
	CRForest::ConstPtr forest_ptr;
	vector<vector<int> > class_ids_;
	Size patch_size_;

public:
	typedef shared_ptr<CRForestDetector> Ptr;
	typedef shared_ptr<CRForestDetector const> ConstPtr;
	
	// Constructor
	CRForestDetector(CRForest::ConstPtr forest_ptr) : forest_ptr(forest_ptr) {
		this->patch_size_ = GODDetectionParams::get()->patch_size;
		forest_ptr->GetClassID(this->class_ids_);
	}

	void detectPyramidMR(vector<Candidate> &candidates, Mat &depth_img, int this_class, float threshold_this_class);
	void calcClassPriors();
	void voteForCandidate(Mat &depth_img, Candidate &new_cand);
	void assignCluster(Mat &img, Mat &depth_img);
	void backprojectCandidate(Mat &img, Candidate &cand);

private:
	void detectPeaks(vector<Candidate> &candidates, bool separate = true, float shift = -1.0f, int this_class = -1);
	void voteColor(Mat &depth_img, float xShift = -1.0f, float yShift = -1.0f, int this_class = -1, Rect &focus = default_rect__, float prob_threshold = -1);


/*************************************************************/

public:
	size_t GetNumLabels() const {
		return forest_ptr->GetNumLabels();
	}
	void GetClassID(vector<vector<int> > &v_class_ids) const {
		forest_ptr->GetClassID( v_class_ids);
	}

	/** returns false if it could not load the hierarchy */
	bool GetHierarchy(vector<HNode> &hierarchy) const {
		return forest_ptr->GetHierarchy(hierarchy);
	}

	CRForest::ConstPtr GetCRForest() const {
		return forest_ptr;
	}
};
