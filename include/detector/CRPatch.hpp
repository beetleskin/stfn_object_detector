#pragma once
#define _copysign copysign

#include "HoG.hpp"

#include <opencv2/core/core.hpp>

#include <vector>
#include <iostream>


using namespace std;
using namespace cv;



// structure for image patch
struct PatchFeature {
	PatchFeature() {}
	Rect roi;
	Point center;
	vector<Mat> vPatch;
};


class CRPatch {
public:
	CRPatch(CvRNG *pRNG, Size patch_size) : cvRNG(pRNG), patch_size(patch_size) {}


	void setClasses(int l) {
		vLPatches.resize(l);
		vImageIDs.resize(l);
	}

	// Extract patches from image
	void extractPatches(Mat &img, Mat &depth_img, unsigned int n, int label, int imageID, Point &object_center);
	// Extract features from image
	static void extractFeatureChannels(Mat &img, Mat &depth_img, vector<Mat> &vImg);

	vector<vector<PatchFeature> > vLPatches;
	vector<vector<int> > vImageIDs;// vector the same size as

private:
	CvRNG *cvRNG;
	Size patch_size;
	static HoG hog;
};

