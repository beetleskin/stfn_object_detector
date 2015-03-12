#include "CRForestDetector.hpp"
#include "GODImageMemory.hpp"
#include "myutils.hpp"

#include "opencv2/gpu/gpu.hpp"

#include <tbb/task_group.h>

#include <vector>


using namespace std;
using namespace cv;



// given the cluster assignment images, we are voting into the voting space GODImageMemory::hough_space
void CRForestDetector::voteColor(Mat &depth_img, float xShift, float yShift, int this_class, Rect &focus, float prob_threshold) {
	// GODImageMemory::hough_space are all initialized before

	if (GODImageMemory::assign_images.size() < 1)
		return;

	// compensate for the shift
	if (xShift < 0)
		xShift = GODImageMemory::assign_images[0].cols * 0.5;
	if (yShift < 0)
		yShift = GODImageMemory::assign_images[0].rows * 0.5;

	float ntrees = float(GODImageMemory::assign_images.size());
	tbb::task_group tbb_tg;

	// loop over trees
	for (size_t trNr = 0; trNr < GODImageMemory::assign_images.size(); ++trNr) {

		function<void ()> process = [ &, trNr]() {
			// loop over assign height
			for (int cy = 0 ; cy < GODImageMemory::assign_images[trNr].rows; ++cy) {
				float *ptr = GODImageMemory::assign_images[trNr].ptr<float>(cy);

				// loop over assign width
				for (int cx = 0; cx < GODImageMemory::assign_images[trNr].cols; ++cx) {
					// get the leaf_id
					if (ptr[cx] < 0)
						continue;
					float depth_scale = depth_img.ptr<float>(cy)[cx];
					if (depth_scale < 0.1f)
						depth_scale = 1.f;

					LeafNode *tmp = forest_ptr->getTrees()[trNr]->getLeaf(ptr[cx]);

					// loop over labels
					for (size_t lNr = 0; lNr < GODImageMemory::hough_space.size(); ++lNr) {

						if ((this_class >= 0 ) && (this_class != lNr)) // the voting should be done on a single class only
							continue;

						bool condition;
						if (prob_threshold < 0) {
							condition = (class_ids_[trNr][lNr] > 0 && tmp->vPrLabel[lNr] * class_ids_[trNr].size() > 1);
						} else {
							condition = (class_ids_[trNr][lNr] > 0  &&  GODImageMemory::class_prior[lNr].ptr<float>(cy)[cx] > prob_threshold);
						}

						if (condition) {
							// vote for all points stored in a leaf
							float w = tmp->vPrLabel[lNr] / ntrees;
							vector<float>::const_iterator itW = tmp->vCenterWeights[lNr].begin();
							for (vector<Point>::const_iterator it = tmp->vCenter[lNr].begin(); it != tmp->vCenter[lNr].end(); ++it, itW++) {
								// calc object hypothesis center
								int x = int(float(cx) - float((*it).x) / depth_scale + 0.5 + xShift);
								int y = int(float(cy) - float((*it).y) / depth_scale + 0.5 + yShift);


								// finally vote into voting space
								if (focus.width == 0) {
									if (y >= 0 && y < GODImageMemory::hough_space[lNr].rows && x >= 0 && x < GODImageMemory::hough_space[lNr].cols) {
										GODImageMemory::hough_space[lNr].ptr<float>(y)[x] += w * (*itW);
									}
								} else if (focus.contains(Point(x, y))) {
									GODImageMemory::hough_space[lNr].ptr<float>(y - focus.y)[x - focus.x] += w * (*itW);
								}
							}
						}
					}
				}
			}
		};

		tbb_tg.run(bind(process));
	}
	tbb_tg.wait();
}


// Gathering the information in the support of each candidate
void CRForestDetector::voteForCandidate(Mat &depth_img, Candidate &cand) {


	//
	// get parameters from world model
	int kernel_width = GODDetectionParams::get()->kernel_vote;
	Size2f max_size = GODDetectionParams::get()->candidate_max_bb;
	CvRNG cvRNG(GODDetectionParams::get()->random_seed);
	bool do_backprojection = GODDetectionParams::get()->do_backprojection;


	double value = 0.0;
	double sample_votes = 0.998;// ignore the patches by this probability
	int min_x, min_y, max_x, max_y;
	cand.backprojection_mask = Mat::zeros(depth_img.size(), CV_32FC1);

	// initializing the box around the candidate center where the votes can come from
	min_x = cand.center.x - int(max_size.width / 2.0f + kernel_width + 0.5f);
	min_y = cand.center.y - int(max_size.height / 2.0f + kernel_width + 0.5f);
	min_x = max(min_x, 0);
	min_y = max(min_y, 0);

	max_x = cand.center.x + int(max_size.width / 2.0f + 0.5f) + 1;
	max_y = cand.center.y + int(max_size.height / 2.0f + 0.5f) + 1;
	max_x = min(GODImageMemory::assign_images[0].cols, max_x);
	max_y = min(GODImageMemory::assign_images[0].rows, max_y);


	// looping over all trees
	float ntrees = float(GODImageMemory::assign_images.size());
	for (size_t trNr = 0; trNr < int(ntrees); trNr++) {

		// looping over all locations within candidate roi
		for (int cy = min_y; cy < max_y; ++cy) {
			float *ptr = GODImageMemory::assign_images[trNr].ptr<float>(cy);

			for (int cx = min_x; cx < max_x; ++cx) {

				value = cvRandReal(&cvRNG);

				if (value < sample_votes || ptr[cx] < 0)
					continue;

				LeafNode *tmp = forest_ptr->getTrees()[trNr]->getLeaf(ptr[cx]);
				float w = tmp->vPrLabel[cand.class_id] / ntrees;
				if (w < 0.0e-7)
					continue;

				float depth_scale = depth_img.ptr<float>(cy)[cx];
				if (depth_scale < 0.1f)
					depth_scale = 1.f;

				float w_element = 0.0f;
				int idNr = 0;
				vector<float>::const_iterator itW = tmp->vCenterWeights[cand.class_id].begin();
				for (vector<Point>::const_iterator it = tmp->vCenter[cand.class_id].begin() ; it != tmp->vCenter[cand.class_id].end(); ++it, ++idNr, itW++) {
					int x = int(float(cx) - float((*it).x) / depth_scale + 0.5);
					int y = int(float(cy) - float((*it).y) / depth_scale + 0.5);

					float squared_dist = (x - cand.center.x) * (x - cand.center.x) + (y - cand.center.y) * (y - cand.center.y);
					if (squared_dist < kernel_width * kernel_width) {
						w_element += w * (*itW);
					}
				}

				if ( w_element > 0.0 && do_backprojection) {
					// update the backprojection image
					cand.backprojection_mask.ptr<float>(cy)[cx] += w_element;
				}
			}
		}
	}
}



/********************************** FULL object detection ************************************/
void CRForestDetector::detectPeaks(vector<Candidate> &candidates, bool separate, float shift, int this_class) {

	//
	// get parameters from world model
	int kernel_width = GODDetectionParams::get()->kernel_smooth;
	int kernel_std = GODDetectionParams::get()->kernel_sigma;
	int max_cands = GODDetectionParams::get()->max_candidates;
	float threshold = GODDetectionParams::get()->detection_threshold;



	candidates.clear();

	// this is just to access a non-empty detect image for getting sizes and so on
	int default_class = 0;
	if ((this_class >= 0) )
		default_class = this_class;

	// smoothing the accumulator matrix
	vector<gpu::GpuMat> smoothAcc;
	smoothAcc.resize(GODImageMemory::hough_space.size());

	
	for (int cNr = 0; cNr < GODImageMemory::hough_space.size(); ++cNr) {
		if ((this_class >= 0) && ( this_class != cNr))
			continue;
		gpu::GpuMat gpu_imgDetect(GODImageMemory::hough_space[cNr]);
		gpu::GaussianBlur(gpu_imgDetect, smoothAcc[cNr], Size(kernel_width, kernel_width), kernel_std);
	}

	// each candidate is a six element vector weight, x, y, scale, class, ratio
	Point max_loc_temp;
	Point min_loc_temp;
	double min_val_temp = 0;
	double max_val_temp = 0;

	float xShift;
	float yShift;

	/***************** find the local maximum locations **********************/
	int candNr = 0;
	for (int count = 0; candNr < max_cands ; ++count) { // count can go until infinity
		bool flag = false;
		Candidate cand;
		// detect the maximum
		if (shift < 0.0f) {
			xShift = GODImageMemory::hough_space[default_class].cols * 0.25;
			yShift = GODImageMemory::hough_space[default_class].rows * 0.25;
		} else {
			xShift = GODImageMemory::hough_space[default_class].cols * shift;
			yShift = GODImageMemory::hough_space[default_class].rows * shift;
		}
		for (size_t cNr = 0; cNr < GODImageMemory::hough_space.size(); ++cNr) {
			if ((this_class >= 0) && ( this_class != cNr))
				continue;

			gpu::minMaxLoc(smoothAcc[cNr], &min_val_temp, &max_val_temp, &min_loc_temp, &max_loc_temp);
			if(max_val_temp >= threshold) {
				flag = true;
				cand.confidence = max_val_temp;
				cand.center.x = float(-xShift + max_loc_temp.x);
				cand.center.y = float(-yShift + max_loc_temp.y);
				cand.class_id = cNr;
			}
		}

		if (!flag)
			break;
		else
			candNr++;

		// push the candidate in the stack
		candidates.push_back(cand);


		// remove the maximum region
		if (shift < 0.0f) {
			xShift = GODImageMemory::hough_space[default_class].cols * 0.25;
			yShift = GODImageMemory::hough_space[default_class].rows * 0.25;
		} else {
			xShift = GODImageMemory::hough_space[default_class].cols * shift;
			yShift = GODImageMemory::hough_space[default_class].rows * shift;
		}

		// remove the region with the supporting kernel width
		int cx = int(cand.center.x + xShift);
		int cy = int(cand.center.y + yShift);
		int x = max(0, cx - kernel_width);
		int y = max(0, cy - kernel_width);
		int rwidth = max(1, min(cx + kernel_width, smoothAcc[default_class].cols - 1) - x + 1);
		int rheight = max(1, min(cy + kernel_width, smoothAcc[default_class].rows - 1)  - y + 1);

		for (int cNr = 0; cNr < GODImageMemory::hough_space.size(); ++cNr) {
			if (cand.class_id >= 0 && cNr != cand.class_id)
				continue;

			// clear candidates bounding box
			smoothAcc[cNr](Rect(x, y, rwidth, rheight)) = Scalar(0.0);
		}
	}
}

void CRForestDetector::detectPyramidMR(vector<Candidate> &candidates, Mat &depth_img, int this_class, float threshold_this_class) {
	

	Mat tmp_depth = depth_img.clone();
	Mat points(tmp_depth.size(), CV_32FC3);
	Mat dist(tmp_depth.size(), CV_32FC1);
	tmp_depth.convertTo(tmp_depth, CV_32FC1);
	calcPoints(tmp_depth, points);
	for (int y = 0; y < points.rows; ++y) {
		Eigen::Vector3f *r_ptr_points = points.ptr<Eigen::Vector3f>(y);
		float *r_ptr_dist = dist.ptr<float>(y);
		for (int x = 0; x < points.cols; ++x) {
			r_ptr_dist[x] = r_ptr_points[x].norm();
		}
	}


	// accumulating votes for this_class
	for (size_t lNr = 0; lNr < forest_ptr->GetNumLabels(); ++lNr) {
		if ( (this_class >= 0 ) && (this_class != lNr) )
			continue;

		// the hough space is twice the size of the input image
		GODImageMemory::hough_space[lNr] = Mat::zeros(GODImageMemory::assign_images[0].size() * 2, CV_32FC1);
	}

	voteColor(dist, -1, -1, this_class, default_rect__, threshold_this_class);


	// detecting the peaks in the voting space
	detectPeaks(candidates, true, -1, this_class);
}



// **********************************    LEAF ASSIGNMENT      ***************************************************** //

// matching the image to the forest and store the leaf assignments in vImgAssing
void CRForestDetector::assignCluster(Mat &img, Mat &depth_img) {
	

	//
	// reset the assignment images
	int nTrees = forest_ptr->getTrees().size();
	GODImageMemory::assign_images.resize(nTrees);
	for (int treeNr = 0; treeNr < nTrees; treeNr++) {
		GODImageMemory::assign_images[treeNr] = Mat(img.size(), CV_32FC1, Scalar(-1.0));
	}

	//
	// extract features
	vector<Mat> features;
	CRPatch::extractFeatureChannels(img, depth_img, features);


	//
	// assign to each pixel its posterior class probability by regressing each tree in the forest

	// x,y top left; cx,cy center of patch
	int xoffset = patch_size_.width / 2;
	int yoffset = patch_size_.height / 2;
	tbb::task_group tbb_tg;

	for (int y = 0; y < img.rows - patch_size_.height; ++y) {

		function<void ()> process = [ &, y]() {

			for (int x = 0; x < img.cols - patch_size_.width; ++x) {
				vector<const LeafNode *> result;

				forest_ptr->regression(result, features, x, y);

				for (size_t treeNr = 0; treeNr < result.size(); treeNr++) {
					GODImageMemory::assign_images[treeNr].ptr<float>(y + yoffset)[x + xoffset] = float(result[treeNr]->idL);
				}

			} // end for x
		};


		tbb_tg.run(bind(process));
		if (y % 50 == 0)
			tbb_tg.wait();


	} // end for y
	tbb_tg.wait();
}


// Getting the per class confidences TODO: this has to become scalable
void CRForestDetector::calcClassPriors() {
	int nlabels = forest_ptr->GetNumLabels();

	// allocating space for the classConfidence
	GODImageMemory::class_prior.resize(nlabels);
	for (int j = 0; j < nlabels; j++) {
		GODImageMemory::class_prior[j] = Mat::zeros(GODImageMemory::assign_images[0].size(), CV_32FC1);
	}


	int w = GODImageMemory::assign_images[0].cols;
	int h = GODImageMemory::assign_images[0].rows;

	// function variables
	int outer_window = 8; // TODO: this parameter shall move to the inputs.
	float inv_tree = 1.0f / GODImageMemory::assign_images.size();

	// looping over the trees
	for (size_t trNr = 0; trNr < GODImageMemory::assign_images.size() ; trNr++) {
		// here make a temporary structure of all the probabilities and then smooth it with a kernel.
		vector<Mat> tmp_class_prior(nlabels);
		for (int cNr = 0; cNr < nlabels; ++cNr) {
			tmp_class_prior[cNr] = Mat::zeros(GODImageMemory::assign_images[trNr].size(), CV_32FC1);
		}

		for (int y = 0; y < h ; ++y) {
			for (int x = 0; x < w; ++x) {
				int leaf_id = GODImageMemory::assign_images[trNr].ptr<float>(y)[x];
				if (leaf_id < 0)
					continue;

				LeafNode *tmp = forest_ptr->getTrees()[trNr]->getLeaf(leaf_id);

				for (int cNr = 0; cNr < nlabels; ++cNr) {
					tmp_class_prior[cNr].ptr<float>(y)[x] = tmp->vPrLabel[cNr] * inv_tree;
				}
			}
		}

		// now values of the tmp_class_prior are set we can blur it to get the average
		for (int cNr = 0; cNr < nlabels; cNr++) {
			blur(tmp_class_prior[cNr], tmp_class_prior[cNr], Size(outer_window, outer_window));
		}

		for (int cNr = 0; cNr < nlabels; cNr++) {
			//  LOOPING OVER ALL PIXELS
			for (int y = 0; y < h; y++) {
				for (int x = 0 ; x < w; x++) {
					GODImageMemory::class_prior[cNr].ptr<float>(y)[x] += tmp_class_prior[cNr].ptr<float>(y)[x];
				}
			}
		}
	}
}



void CRForestDetector::backprojectCandidate(Mat &img, Candidate &cand) {

	Mat mask_thresh;
	float thresh = 2;
	int blur_kernel_size = 9;



	// smooth backprojection mask
	blur(cand.backprojection_mask, cand.backprojection_mask, Size(blur_kernel_size, blur_kernel_size));

	// get the maximum and minimum values in the backprojection mask
	double min_val_temp = 0;
	double max_val_temp = 0;
	minMaxLoc(cand.backprojection_mask, &min_val_temp, &max_val_temp);

	// determine the threshold
	double thresh_val = float(thresh) * (min_val_temp + max_val_temp) / 20.0f ; // TODO: make this a separate parameter

	// thresholding the image
	threshold(cand.backprojection_mask, mask_thresh, thresh_val, 1, CV_THRESH_BINARY);

	// now we have to determine the bounding box
	int min_x = cand.backprojection_mask.cols;
	int min_y = cand.backprojection_mask.rows;
	int max_x = -1;
	int max_y = -1;

	for (int y_ind = 0; y_ind < mask_thresh.rows ; ++y_ind) {
		float *ptr = mask_thresh.ptr<float>(y_ind);
		for ( int x_ind = 0; x_ind < mask_thresh.cols; ++x_ind) {
			if (ptr[x_ind] > 0) {
				if (y_ind > max_y)
					max_y = y_ind;
				if (x_ind > max_x)
					max_x = x_ind;
				if (x_ind < min_x)
					min_x = x_ind;
				if (y_ind < min_y)
					min_y = y_ind;
			}
		}
	}

	cand.bb[0].x = min_x;
	cand.bb[0].y = min_y;
	cand.bb[1].x = max_x;
	cand.bb[1].y = max_y;

	cand.backprojection_mask.release();
}